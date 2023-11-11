############################################################################################################################
############################################--- Hubbard Model by DFT ---####################################################
############################################################################################################################


############################################################################################################################
####################################################--- Importing ---#######################################################
############################################################################################################################

using LinearAlgebra  #Pacote de Algebra linear
using Base.Threads      # numbers in binary representation
using LaTeXStrings  # For LaTex code
using SpecialFunctions  #Special Functions
using Integrals  #Integrals

############################################################################################################################
#############################################--- Setting the parameters ---################################################
############################################################################################################################

const U=2.0                        #Couloumb repultion
const L=2                          #Chain size
      v=zeros(L)                    #Site energy
const ΔE=1.E-8                     #Convergence parameter
const nj_0=0.5*ones(L)             #Intial density
const N=Int(L/2)                    #Number of electrons
const file_="0"                     # 1 write an article, 0 does not 
println("number of threads= ",Threads.nthreads())

v[1]=0   
v[2]=0                                 #Impurity energy


############################################################################################################################
#################################################---- Functions ----########################################################
############################################################################################################################

############################################---- BA fundamental energy ----#####################################################

function integ(U,x)

    f=[]

    for i in x

        push!(f,besselj0(i)*besselj1(i)/(i*(1+exp(U*i/2))))

    end

    return f
end

function Simp_rule(a,b,values,dx)

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
    
    return -4*Simp_rule(a,b,f_x,dx)
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

function mH(L,U,βU,nj_,vj,μ,N)

    mat=zeros(L,L)
    vKS=Pot_KS(nj_,βU,U)
    vH=Pot_H(nj_,U)

    for i in 1:1:L-1

        mat[i,i+1]=mat[i+1,i]+=-1.0

    end

    for i in 1:1:L

        mat[i,i]+=vKS[i]+vj[i]+vH[i]+μ

    end

    return mat
end

#############################################---- Eletronic Density----#####################################################

function nj(Vec,N,L)

    nj_=zeros(L)

    for j in 1:1:L

        for i in 1:N

            nj_[j]+=abs2(Vec[j,i])

        end
    end

    return nj_
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

############################################################################################################################
#########################################---- Main Calculation function ----################################################
############################################################################################################################

function SCC(L,N,U,ΔE,v,nj_0)

    E0_BA=BA_E0L(U)
    βU=bU(E0_BA)
    E0=E_fun(nj_0,βU,U)/L

    H=mH(L,U,βU,nj_0,v,0.0,N)
    Ener=eigvals(H)
    Vec=eigvecs(H)

    nj_=2*nj(Vec,N,L)
    E=E_fun(nj_,βU,U)/L
    μ=-Ener[N+1]
    n=0

    while abs(E-E0)>=ΔE

        nj_0=(0.95.*nj_0.+0.05.*nj_)
        E0=E

        H=mH(L,U,βU,nj_0,v,μ,N)
        Ener=eigvals(H)
        Vec=eigvecs(H)
        μ=-Ener[N+1]
        nj_=2*nj(Vec,N,L)
        println(nj_)
        E=E_fun(nj_,βU,U)/L

        println("Step Energy= ",E," ","BA Energy= ",E0_BA," ","DIF.=",abs(E-E0_BA)," μ=",μ)
        n+=1

    end

    println("Step Energy= ",E," ","BA Energy= ",E0_BA," DIF.=",abs(E-E0_BA)," ","μ=",μ)

    return E, nj_

end


@time ener,den=SCC(L,N,U,ΔE,v,nj_0)


if file_=="1"

    file=open("Imp_DFT_MHU"*string(U)*"L"*string(L)*"N"*string(N)*".txt","w")  # file's name -> W value, r means random, 1 is the delta's value and 100 is the number of points of time, essemble's size
       
    
    write(file,"F_Energy"," ",string(ener),"\n")

    for i=1:1:L
        
        write(file,string(i)," ",string(den[i]),"\n")
                
    end
        
    close(file)

end