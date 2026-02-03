function [eleccion, indice]=Fun_VIKOR(m,n,V,w,MM)

%   4/06/2017
%   Ufa State Petroleum Technological University
%   Author: Dr. Irik Z. Mukhametzyanov
%   
%   This Matlab code file is provided for VIKOR implimentation function code  
%   (VI?ekriterijumso KOmpromisno Rangiranje)

%   References:
%   Opricovic, S., Tzeng, G. H., “Compromise solution by MCDM methods: A comparative analysis of VIKOR and TOPSIS,
%   European Journal of Operational research, Vol. 156, No. 2, pp. 445-455, 2004.

% Si strategy of the maximum group utility    
% Ri strategy of "the majority of criteria"    
% Qi compromise solution
%    
% - Before executing the function you have to define DMM-Decision Making
%   Matrix [mxn] variable based on size of decision making matrix that 
%   you have.
%   w - the weight of attributes determined by decision-maker [1xn]
%   MM - criteriaSign [1xn] matrix; =1 for benefit (revenue) attributes доход; 
%                                   =-1 for cost attributes (expenditure) расход
%   v  is introduced as weight of the strategy of "the majority of attributes"
%     "Voting by majority rule" (v> 0.5); 
%        or "by consensus" (for v = 0.5); 
%          or "with a veto" (for v <0.5).
%--------------------------------------------------------------------------
   
   %-- Standart VIKOR
   v=0.5;
   V=Fun_DMnorm(m,n,V,MM,3); 
   
   Vmax=max(V);
   Vmin=min(V);
   %-- Determine Ideal (PIS) and Negative-Ideal (NIS) Solutions
   for j=1:n
       if MM(j)==1
           PIS(j)= Vmax(j);
           NIS(j)= Vmin(j);
       else
           PIS(j)= Vmin(j);
           NIS(j)= Vmax(j);  
       end
   end
   
   %-- the "closest" to the ideal
   dV1=( repmat(PIS,m,1) - V) ./ repmat(PIS-NIS,m,1);
  % dV2=( V - repmat(NIS,m,1)) ./ repmat(PIS-NIS,m,1);
  
   dV1=dV1.*repmat(w,m,1);
  % dV2=dV2.*repmat(w,m,1);
   
   S=zeros(m,1); R=zeros(m,1); C=zeros(m,1); Rnk1=zeros(m,1);

    S=sum(dV1,2);              % L1-metric 
    for i=1:m
        R(i)=max(dV1(i,:));    % Linf-metric
    end

    %-------   S+=S1, S-=S2
    S1=min(S);    S2=max(S);     
    R1=min(R);    R2=max(R);
 
    C=v*(S-S1)/(S2-S1)+(1-v)*(R-R1)/(R2-R1); 

   %-- ranking:  MIN the best
    [Q,ix] =sort(C);     % сортировка в порядке возрастания (if 'descend' убывания), ix-индексы
   
   Sx=zeros(m,1); Rx=zeros(m,1);
   for i=1:m
        Sx(i)=S(ix(i));
        Rx(i)=R(ix(i));
   end  
    
  
 %============================ Condition ==================================
 % C1:  C(A2)-C(A1) >= DQ=1/(m-1)    Допустимое преимущество  =>
 % C2:  S(A2)-S(A1) >=DQ  .or. R(A2)-R(A1) >= DQ   
 % Qmin the best 
 %---------------------------------------------------
  
 DQ=1/(m-1);
 Condition1=0; Condition2=0;
                
  if Q(2)-Q(1) >= DQ
       Condition1=1;
  end
  if S(ix(2))-S(ix(1)) >= DQ  |  R(ix(2))-R(ix(1)) >= DQ
      Condition2=1;
  end 
 
  k=1;
  Rank=tiedrank(C);
  
    if Condition1 & Condition2        
        %  'A1 '
    end
    if Condition1 & ~Condition2          
        %  'A1 = A2'
        Rank(ix(2))=1;
    end
    if ~Condition1         
         for i=2:m-1
             if Q(i)-Q(1) < DQ  &  Q(i+1)-Q(1) >= DQ
                  k=i;
                  break
             end 
         end
         %  'A1 = A2 =... = Ak'
         for i=2:k   % различимость рангов (первые k альтернатив не различимы)
             Rank(ix(i))=1;
         end      
    end               

	[eleccion, indice] = min(Q);

return