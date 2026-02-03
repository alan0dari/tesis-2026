function [eleccion, indice]=topsis(m,n,DM,w,MM)

   R=zeros(m,n);
   V=zeros(m,n);
   
   PIS=zeros(1,n);
   NIS=zeros(1,n);
   
   R=Fun_DMnorm(m,n,DM,MM,3);  
   V=R.*repmat(w,m,1);
   
   for i=1:n
       if MM(i)==1
           PIS(i)=max(V(:,i));
           NIS(i)=min(V(:,i));
       else
           PIS(i)=min(V(:,i));
           NIS(i)=max(V(:,i));  
       end
   end
   
   ED1=sqrt(sum((repmat(PIS,m,1)-V).^2,2));
   ED2=sqrt(sum((repmat(NIS,m,1)-V).^2,2));
   
   C=ED2./(ED1+ED2);
   [eleccion, indice] = max(C);
            
return