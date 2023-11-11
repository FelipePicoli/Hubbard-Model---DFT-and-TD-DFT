using LinearAlgebra  #Pacote de Algebra linear
using Random, Distributions  # random number's distributions
using BitBasis,Base.Threads      # numbers in binary representation
using LaTeXStrings  # For LaTex code
using SpecialFunctions  #Special Functions
using Integrals  #Integrals


Δ=-0.5
U=0.0
ti=0.0
tf=10.0
J=5.0
N=10000

#Potential

function W_p(w,J,t) 

    v=w*t/J
    
    if abs(v)>=abs(w)

        v=w      # Potential null for t<0

    elseif t<0

        v=0

    end
    
    return v

end

#Matrix

function mH(Δ1,Δ2,U)

    m=zeros(6,6)

    m[1,1]+=2*Δ1+U
    m[2,2]+=Δ1+Δ2
    m[3,3]+=Δ1+Δ2
    m[4,4]+=Δ1+Δ2
    m[5,5]+=Δ1+Δ2
    m[6,6]+=U+2*Δ2
    m[1,2]=m[2,1]+=-1
    m[1,3]=m[3,1]+=-1
    m[2,6]=m[6,2]+=-1
    m[3,6]=m[6,3]+=-1

    return m
end

#Occupation number

function nl(Vec)

    site1=[2 1 1 1 1 0]
    site2=[0 1 1 1 1 2]

    n=zeros(2)

    for j in 1:1:6
        
        n[1]+=site1[j]*abs2(Vec[j])
        n[2]+=site2[j]*abs2(Vec[j])

    end

    return n
end


function main(Δ,U,ti,tf,N,J)

    Tau=LinRange(ti,tf,N)
    n=zeros(2,N)
    dt=(tf-ti)/N

    Vec0=eigvecs(mH(0,0,U))
    n[:,1].+=nl(Vec0[:,1])
    Pi=Vec0[:,1]

    for i in 2:N
        
        H=mH(W_p(Δ,J,Tau[i]),-1.0*W_p(Δ,J,Tau[i]),U)
        Vec=eigvecs(H)
        Ener=eigvals(H)
        Pip1=exp(-1im*H*dt)*Pi
        n[:,i].+=nl(Pip1)
        println(nl(Vec[:,1]))
        #println(nl(Pip1))
        Pi=Pip1

    end

    
    return Tau,n
end

t,n=main(Δ,U,ti,tf,N,J)

file=open("ExatoU"*string(U)*"Delta"*string(Δ)*"J"*string(J)*".txt","w") 
    

for i=1:1:N
    
    write(file,string(t[i])," ",string(n[1,i])," ",string(n[2,i]),"\n")

end

close(file)