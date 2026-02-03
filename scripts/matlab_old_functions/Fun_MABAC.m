function [eleccion, indice]=Fun_MABAC(m,n,DM,w,MM)

%****************  <MCDM> Data Analis -------------------------------------
%   13/10/2017
%   Ufa State Petroleum Technological University
%
%   Author: Dr. Irik Z. Mukhametzyanov (e-mail: izmukhametzyanov@gmail.com )
%
%--------------------------------------------------------------------------
%     MABAC Multi-Attributive Border Approximation area Comparison         
%--------------------------------------------------------------------------  
% 
%   This Matlab code file is designed to solve MCDM problems
%
%   References:
%   Dragan Pamuñar & Goran Cirovic 
%   The selection of transport and handling resources in logistics centres 
%   using Multi-Attributive Border Approximation area Comparison (MABAC)
%   SJM #1 2017


% - Before executing the function you have to define DMM-Decision Making
%   Matrix [mxn] variable based on size of decision making matrix that 
%   you have.

%   w - the Rank of kriteria determined by decision-maker [1xn]
%   MM - criteriaSign [1xn] matrix; =1 for benefit (revenue) attributes; 
%                                   =-1 for cost attributes (expenditure)
           
    
%----------------------- MABAC Mehtod  --------------------------


    D1=ones(m,n);
    
    %-- Construct the Weighted Normalized Decision Matrix with
    %   Determine Ideal  and Negative-Ideal  Solutions
    
	V=Fun_DMnorm(m,n,DM,MM,3); 
    V=(DM+D1).*repmat(w,m,1);     
   
    g=geomean(V);
    Q=V-repmat(g,m,1);
    C=sum(Q,2);
	
	[eleccion, indice] = max(C);

return