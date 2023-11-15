############################################################################################################################
###########################################--- Photoemission Ramp-up ---####################################################
############################################################################################################################

############################################################################################################################
####################################################--- Importing ---#######################################################
############################################################################################################################

using LinearAlgebra  #Pacote de Algebra linear
using Random, Distributions  # random number's distributions
using BitBasis,Base.Threads      # numbers in binary representation
using LaTeXStrings  # For LaTex code
using SpecialFunctions  #Special Functions
using Integrals  #Integrals

############################################################################################################################
#############################################--- Setting the parameters ---################################################
############################################################################################################################

const U=5.0                        #Couloumb repultion
const L=2                          #Chain size
      v0=zeros(L)                    #Site energy
      v=zeros(L)                    #Site energy
const ΔE=1.E-6                     #Convergence parameter
const nj_0=0.5.*ones(L)             #Intial density
const N=Int(L/2)                    #Number of electrons
const file_="1"                     # 1 write an article, 0 does not 
const t_i=0.0                       # Intial Time
const t_f=10                     # Final time
const N_time=1000                  # Number of time steps
const W=-0.5                       # Scatering potential
const SP_site=1                   # Scatering potential Site
const Exc=100
const J=5.0

println("number of threads= ",Threads.nthreads())

############################################################################################################################
#################################################---- Functions ----########################################################
############################################################################################################################


############################################---- Potential Function ----#####################################################

function W_p(w,J,t) 

    v=w*t/J
    
    if abs(v)>=abs(w)

        v=w      # Potential null for t<0

    elseif t<0

        v=0

    end
    
    return v

end

############################################---- BA fundamental energy ----#####################################################

function integ(U,x)

    f=[]

    for i in x

        push!(f,besselj0(i)*besselj1(i)/(i*(1+exp(U*i/2))))

    end

    return f
end

function Simp_rule(values,dx)

    s=values[1]+values[length(values)]

    for i in 2:1:length(values)-1

        if i%2==0
            s+=values[i]*4
        else
            s+=values[i]*2
        end

    end

    return s*dx/3

end

function BA_E0L(U)

    a=10^-8
    b=10^5
    x=LinRange(a,b,10^7)
    f_x=integ(U,x)
    dx=x[2]-x[1]
    
    return -4*Simp_rule(f_x,dx)
end

#################################################---- Finding β(U)----#####################################################

function f(x)

    return -(2*x/pi)*sin(pi/x)
    
end

function df(x)

    y=x/pi

    return (2*cos(1/y)/y-2*sin(1/y))/pi
    
end

function NM(x0,E0)

    rpi2=floor(E0*2/pi)

    x0=x0+rpi2*pi/2

    xnl=x0
    xn=xnl-(f(xnl)-E0)/df(xnl)
    N=0

    while (abs(xnl-xn)>10^-6)&&(N<10^3)

        xnl=xn
        xn=xnl-(f(xnl)-E0)/df(xnl)
        N+=1

    end

    return xn
end

function bU(E0)

    return NM(0.01,E0)
end

#############################################---- Khon-Sham potential----#####################################################

#=
function Pot_KS(nj,βU,U)

    vj=zeros(length(nj))

    for i in 1:1:length(nj)

        n=nj[i]

        if (n>=0)&&(n<1)

            vj[i]+=2*cos(pi*n/2)-U/2*n-2*cos(n*βU)

        elseif n==1

            vj[i]+=0.0

        elseif (n>1)&&(n<=2)

            vj[i]+=-2*cos(pi*(2-n)/2)-U/2*(2-n)+2*cos((2-n)*βU)

        end
    end

    return vj
end=#

function Pot_KS(nj,βU,U)

    vj=zeros(length(nj))

    for i in 1:1:length(nj)

        n=nj[i]

        if (n>=0)&&(n<=1)

            vj[i]+=-2*cos(pi*n/2)-U/2*n-2*cos(n*pi/βU)

        elseif (n>1)&&(n<=2)

            vj[i]+=2*cos(pi*(2-n)/2)-U/2*n+2*cos((2-n)*pi/βU)-2*βU*U/pi

        end
    end

    return vj
end

function Pot_H(nj,U)

    vj=zeros(length(nj))

    for i in 1:1:length(nj)

        n=nj[i]

        vj[i]+=U*n/2

    end

    return vj
end

#############################################---- Hamiltonian construction----#####################################################

function mH(L,U,βU,nj_,vj,μ)

    mat=zeros(L,L)
    if U==0
        vKS=zeros(L)
    else
        vKS=Pot_KS(nj_,βU,U)
    end
    
    vH=Pot_H(nj_,U)

    for i in 1:1:L-1

        mat[i,i+1]=mat[i+1,i]+=-1.0

    end

    for i in 1:1:L

        mat[i,i]+=vKS[i]+vj[i]+vH[i]+μ

    end

    return mat
end

#############################################---- Fundamental State Energy----#####################################################

function E_fun(nj_,βU,U)

    E=0.0

    for n in nj_

        if (n>=0)&&(n<=1)

            E+=sin(pi*n/βU)

        elseif (n>1)&&(n<=2)

            E+=sin(pi*(2-n)/βU)+U*(n-1)*pi/(-2*βU)

        end
    end

    return -2*βU*E/pi
end

#############################################---- Mix Desnity----#####################################################

function Mix(n1,n2)

    n=zeros(length(n1))

    for i in 1:1:length(n1)-1

        n[i]+=n1[i]+n2[length(n2)-i]

    end

    return 0.5.*n

end

##########################################---- Calculating the final states ----##############################################

function list_return(vec)    # Convert  list of bools to a list of floats

    v=[]

    for x in vec
        if x==false
            push!(v,0.0)
        else
            push!(v,1.0)
        end
    end

    return v
end
function sum_half(vec)

    size_=length(vec)
    s=0

    for i=Int(size_/2+1):1:size_
        s+=vec[i]
    end

    return s
end

function FinStates(L,N,Exc)  #number of electrons
    
    posible=[]
    
    for numb in 1:1:(2^(L)-1)
        
        Binary_rep=bitarray(numb,L)

        if sum(Binary_rep)==N && sum_half(Binary_rep)<=Exc
            push!(posible,Binary_rep)
        end
        
    end
    
    return posible
end


##########################################---- Calculating the the intial state ----##############################################

function Initial_State_det(L,N)
    
    v=zeros(L)

    for i in 1:1:length(v)

        if i<=Int(N)
            v[i]+=1.0
        end

    end

    return v
end

##########################################---- Calulating the projection ----##############################################

function Proj(vec1,vec2,matProd)  # first and second vectors, matrix if the eigenvectors projections
    
    Nsize=Int64(sum(vec1))
    mat=zeros(Nsize,Nsize) # define the matrix to calculate the determinant
    
    posiVec1=findall( x -> x == 1,vec1)       # find the ones in the vec1
    posiVec2=findall( x -> x == 1,vec2)       # find the ones in the vec2  
    
    k=1
    for i in posiVec1
        l=1
        for j in posiVec2
            
            mat[k,l]=matProd[i,j]
            l+=1
            
        end
        k+=1
    end

    return det(mat)
end


#######################################---- Calculating the state energy ----##############################################

function StateEnergy(vec,EnerFin)

    return transpose(vec)*EnerFin

end

#######################################---- Calculating the total projection ----##############################################

function ProjTot(VecIni,EnerFin,VecFin,InitialState,Finalstates)  #Initial and Final eigenstates, Initial State, Final States
    
    matProd=transpose(VecIni)*VecFin    # The matrix of the eigenstates produtes
    N_fs=
    ener=zeros(length(Finalstates))
    mat_proj=zeros(length(Finalstates),length(Finalstates))
    mat_proj_h=zeros(length(Finalstates),length(Finalstates))
   

    @threads for p in 1:1:length(Finalstates)

        State_ini=Finalstates[p]
        ener[p]+=StateEnergy(State_ini,EnerFin)          # we are calculating the state energy

        @threads for q in 1:1:length(Finalstates)
       
            State_fin=Finalstates[q]
            a=Proj(State_ini,State_fin,matProd)
            mat_proj[p,q]+=a    # we are calculating the states projections
            mat_proj_h[q,p]+=a

        
        end
    end
    
    return ener, mat_proj, mat_proj_h
end


#######################################---- Calculating the total projection ----##############################################

function TR(Ψ_t,Ψ_0)

    return tr(Ψ_0*conj(transpose(Ψ_0))*Ψ_t*conj(transpose(Ψ_t)))
end

#######################################---- Calculating the total trnasition rate ----##############################################

function expE(dt,ener)

    e=zeros(length(ener),length(ener)).+1im.*zeros(length(ener),length(ener))

    for i in 1:1:length(ener)

        e[i,i]+=exp(-1im*dt*ener[i])

    end

    return e
end

function nj(Vec,N,L)

    nj_=zeros(L)

    for j in 1:1:L

        for i in 1:N

            nj_[j]+=abs2(Vec[j,i])

        end
    end

    return nj_
end

#############################################---- TSelf-Consistent Calculation----#####################################################

function SCC(L,N,U,ΔE,v,nj_0)

    E0_BA=BA_E0L(U)
    βU=bU(E0_BA)

    H=mH(L,U,βU,nj_0,v,0.0)
    Ener=eigvals(H)
    Vec=eigvecs(H)

    nj_=nj(Vec,N,L)

    E0=E_fun(nj_0,βU,U)/L
    E=E_fun(nj_,βU,U)/L
    μ=-Ener[N+1]
    n=0
    
    while sum(abs.(nj_-nj_0))/L>=ΔE

        nj_0=(0.95.*nj_0.+0.05.*nj_)
        E0=E

        H=mH(L,U,βU,nj_0,v,μ)
        Ener=eigvals(H)
        Vec=eigvecs(H)
    
        nj_=nj(Vec,N,L)
    
        E=E_fun(nj_,βU,U)/L
        μ=-Ener[N+1]
        n+=1

    end

    return E, nj_, mH(L,U,βU,nj_,v,μ), eigvecs(mH(L,U,βU,nj_,v,μ)), eigvals(mH(L,U,βU,nj_,v,μ))

end

############################################################################################################################
#########################################---- Main Calculation function ----################################################
############################################################################################################################

function distri(w,L,N,Initial_State,FinalStates,t,Ψ_0,J,N_sites,v0,v)
    

    dt=abs(t[1]-t[2])
    res=zeros(N_time)
    res_inst=zeros(N_time)
    on_l=zeros(L,N_time)
    on_l_fund=zeros(L,N_time)
    Ψ_t_m=Ψ_0
    Ψ_t_1=Ψ_0
    absc_0=zeros(length(t))

    E0,nj0,matH_i,Vec0, Ener0=SCC(L,N,U,ΔE,v0,nj_0)

    Vec=1.0.*Vec0
    Ener=1.0.*Ener0
    EnerFin=1.0.*Ener
    VecFin=1.0.*Vec
    nj=nj0
    
    for t_i in 1:1:length(t)

        absc_0[t_i]+=abs(Ψ_t_1[1])
        on_l_fund[:,t_i].+=n_l_fund(N_sites,VecFin,Initial_State)
        

        if t[t_i]<=abs(J)

            v[1]=W_p(w,J,t[t_i]+dt/2)
            v[2]=-1.0*W_p(w,J,t[t_i]+dt/2)
            
            E,nj,matH_f,Vec,Ener=SCC(L,N,U,ΔE,v,nj0)

            println(v[1])
        
        end

        println(t[t_i])
        VecFin=Vec
        EnerFin=Ener

        ener,matproj,matproj_h=ProjTot(Vec0,EnerFin,VecFin,Initial_State,FinalStates)
       
        Ψ_t_1=expE(dt,ener)*matproj_h*Ψ_t_m

        res_inst[t_i]+=abs2(Ψ_t_1[1])

        Ψ_t=matproj*Ψ_t_1
        
        res[t_i]+=abs2(Ψ_t[1])

        on_l[:,t_i].+=n_l(N_sites,Ψ_t,Vec0,FinalStates)
        println(on_l[:,t_i])

        Ψ_t_m=Ψ_t
               
    end

    return res, res_inst, on_l, on_l_fund, absc_0 
 
end


###########################################---- Irene's Distance ----###############################################

function Distance(som)

    return sqrt(2).*sqrt.(1.0.-abs.(som))
end

##############################################---- Occupation Number ----##############################################

function n_posi(state,posi)

    s=0

    for i in 1:1:posi-1

        s+=state[i]

    end

    return s
end

function conv(vec)

    v=[]

    for i in vec

        push!(v,i)

    end

    return v
end

function n_l(N_sites,Ψ_t,VecFin,FinalStates)

    som=zeros(N_sites).+1im.*zeros(N_sites)

    for p in 1:1:length(FinalStates)

        for q in 1:1:length(FinalStates)

            dif=abs.(FinalStates[p].-FinalStates[q])
            level=findall( x -> x == 1, dif)

            if length(level)==2

                const_p=findall( x -> x == 2,dif.+conv(FinalStates[p]))
                const_q=findall( x -> x == 2, dif.+conv(FinalStates[q]))

                n_p=n_posi(FinalStates[p],const_p[1])
                n_q=n_posi(FinalStates[q],const_q[1])

                for l in 1:1:N_sites

                    som[l]+=Ψ_t[p]*conj(Ψ_t[q])*VecFin[l,level[1]]*VecFin[l,level[2]]

                    
                end
            end
        end

        level=findall( x -> x == 1,FinalStates[p])

        for m in level

            for l in 1:N_sites

                som[l]+=abs2(Ψ_t[p]*VecFin[l,m])

            end
        end
    end
 
    return abs.(som)
    
end 

function n_l_fund(N_sites,VecFin,InitialState)

    som=zeros(N_sites)

    level=findall( x -> x == 1,InitialState)

    for l in 1:1:N_sites

        for m in level

            som[l]+=abs(VecFin[l,m])^2

         end
    end

    return abs.(som)
end 

##############################################---- Distances ----##############################################

function Trace(c_0)

    return sqrt.(abs.(1.0.-abs.(c_0).^2))
end

function Bures(c_0)

    return sqrt(2).*sqrt.(abs.(1.0.-abs.(c_0)))
end

function Den(n_l,n_l_fund,N_sites,N_time)

    s=zeros(N_time)

    for j in 1:1:N_time

        for i in  1:1:N_sites

            s[j]+=2.0*abs(n_l[i,j]-n_l_fund[i,j])/N_sites

        end
    end

    return s
end

############################################################################################################################
#########################################---- Main Calculation function ----################################################
############################################################################################################################

function Main_Calc_func(N,t_i,t_f,N_time,W,Exc,J,v0,v)
    
    Tau=LinRange(t_i,t_f,N_time)
    som_tot=zeros(N_time)
    som_tot_inst=zeros(N_time)
    N_sites=L
    oN_l=zeros(N_sites,N_time)
    oN_l_fund=zeros(N_sites,N_time)
    c_0=zeros(N_time)

    
    Initial_State=Initial_State_det(L,N)  # Initial state in decimal representation
    FinalStates=FinStates(L,N,Exc)  # Possible final states in decimal representation 
               
    Ψ_0=zeros(length(FinalStates))
    Ψ_0[1]+=1.0

   
    a,b,c,d,e=distri(W,L,N,Initial_State,FinalStates,Tau,Ψ_0,J,N_sites,v0,v)
    som_tot.+=a
    som_tot_inst.+=b
    oN_l.+=c
    oN_l_fund.+=d
    c_0.+=e

    println("\n------------------------------------------------------------------\n")

    return Tau, som_tot, som_tot_inst, oN_l, oN_l_fund, Trace(c_0), Bures(c_0), Den(oN_l,oN_l_fund,N_sites,N_time) 

end

############################################################################################################################
#################################################---- Really Calculation ----###############################################
############################################################################################################################

@time t,p,inst,a,b,Tr,Bu,Dens=Main_Calc_func(N,t_i,t_f,N_time,W,Exc,J,v0,v)


if file_=="1"

    file=open("TDDFT"*"Delta"*string(W)*"U"*string(U)*"J"*string(J)*"t_i"*string(t_i)*"t_f"*string(t_f)*"N_time"*string(N_time)*".txt","w")  # file's name -> W value, r means random, 1 is the delta's value and 100 is the number of points of time, essemble's size
       
    #file=open("4te.txt","w")  # file's name -> W value, r means random, 1 is the delta's value and 100 is the number of points of time, essemble's size
    
    #write(file,"Limit"," ",string(c),"\n")

    for i=1:1:N_time
        
        write(file,string(t[i])," ",string(p[i])," ",string(inst[i])," ")

        
        for j in 1:1:L

            write(file, string(a[j,i])," ",string(b[j,i])," ")

        end

        write(file,string(Tr[i])," ",string(Bu[i])," ",string(Dens[i])," ",string(W_p(W,J,t[i])),"\n")
                
    end
        
    close(file)

end