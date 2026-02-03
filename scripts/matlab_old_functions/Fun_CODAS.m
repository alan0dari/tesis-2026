function [eleccion, indice]=Fun_CODAS(m,n,V,w,MM)

%   6/02/2018
%   Ufa State Petroleum Technological University
%   Author: Dr. Irik Z. Mukhametzyanov
%   
%   This Matlab code file is provided for CODAS implimentation function code  
%   (COmbinative Distance-based ASsessment) method

%   References:
%   Kechavarz Ghorabaee, M., Zavadskas, E.K., Turskis, Z. & Antucheviciene,
%   J. (2016). A new COmbinative Distance-based ASsessment (CODAS) method
%   for Multi Criteria  Decision-Making. Economics Computation & Economicc
%   Cybernetics Studies & Research, 50.


% - Before executing the function you have to define DMM-Decision Making
%   Matrix [mxn] variable based on size of decision making matrix that 
%   you have.
%   w - the weight of attributes determined by decision-maker [1xn]
%   MM - criteriaSign [1xn] matrix; =1 for benefit (revenue) attributes äîõîä; 
%                                   =-1 for cost attributes (expenditure) ðàñõîä

        
    %-- Construct the Weighted Normalized Decision Matrix  
    %-- Construct the Weighted Normalized Decision Matrix  norm(D(:,j),p);   
    
    V=Fun_DMnorm(m,n,V,MM,3);    
    V=V.*repmat(w,m,1);
	u=0.03
    
 %-- CODAS method
 
 %-- Determine Negative-Ideal Solutions (NIS) 
    NIS=min(V);

    dV=abs( repmat(NIS,m,1) - V);
    T=sum(dV,2);               %  Taxi Cab Distance
    E=sqrt( sum(dV.^2, 2) );   %  Euclidean Distance 
 
    for i=1:m
        for k=1:m
            s=0;
            if abs(E(i)-E(k))>=u
                s=1;
            end
            H(i,k)=(E(i)-E(k))+s*(T(i)-T(k));
        end
    end
    
    Q=sum(H,2);    %  sum rows
    [eleccion, indice] = max(Q);    %  max- the best alternatives
				   
				
   
return