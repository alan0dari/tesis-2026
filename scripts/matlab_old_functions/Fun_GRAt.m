function [eleccion, indice]=Fun_GRAt(m,n,V,w,MM)

%   05/06/2017 
%   ((update 30/05/2019   +  Normalization of decision matrix DM
%        (2)- Vector method; (3)- Max-Min method; (4)- Sum method
%
%   Ufa State Petroleum Technological University
%   Author: Dr. Irik Z. Mukhametzyanov
%   
%   This Matlab code file is provided for GRA implimentation function code  
%   (Grey Relation Analysis)

%   References:
%    Deng, J.L. (1989). Introduction to grey system. 
%       Journal of Grey System, 1(11), 1-24


% - Before executing the function you have to define DMM-Decision Making
%   Matrix [mxn] variable based on size of decision making matrix that 
%   you have.
%   w - the weight of attributes determined by decision-maker [1xn]
%   MM - criteriaSign [1xn] matrix; =1 for benefit (revenue) attributes äîõîä; 
%                                   =-1 for cost attributes (expenditure) ðàñõîä


    
    C=zeros(m,1); r1=zeros(1,n); r2=zeros(1,n); k=0.5;
    
    %----------------------------- GRA(T) -------------------------------- 
    
    %- Construct the Weighted Normalized Decision Matrix  without W
    %-- V=norm(a)  without Invers
         
    for j=1:n
        if MM(j)==1
           r1(j)=max(V(:,j));
           r2(j)=min(V(:,j));
        end
        if MM(j)==-1
           r1(j)=min(V(:,j)); 
           r2(j)=max(V(:,j));
        end
    end
    A1=abs(repmat(r1,m,1) -V);
    A2=abs(repmat(r2,m,1) -V);
    
    for i=1:m
        S1(i)=max(A1(i,:));
        S2(i)=min(A1(i,:));
        P1(i)=max(A2(i,:));
        P2(i)=min(A2(i,:));
    end
    

    E1=(min(S2)+k*max(S1)) ./ (A1+k*max(S1));
    E2=(min(P2)+k*max(P1)) ./ (A2+k*max(P1));
  % 'E1 E2'
  %  [E1 E2]
    E1= E1.*repmat(w,m,1);
    E2= E2.*repmat(w,m,1); 

    
    G1=sum(E1, 2);
    G2=sum(E2, 2);
    
    C=G1./G2;
	
	[eleccion, indice]=max(C);
    
   % 'G1 G2 G1/G2'
   % [G1 G2 C]
   % max(C) is the best alternatives 

return