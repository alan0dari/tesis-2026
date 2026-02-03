function [pareto, pname] = mopso_clahe2(impath,tmax)
%MOPSOCLAHE FunciÃ³n que devuelve un frente pareto haciendo la mejora de
%contraste por el mÃ©todo MOPSO-CLAHE presentado en "PARAMETER TUNING OF
%CLAHE BASED ON MULTI-OBJECTIVE OPTIMIZATION TO ACHIEVE DIFFERENT CONTRAST LEVELS IN MEDICAL IMAGES"
%Luis G. More, Marcos A. Brizuela, Horacio Legal Ayala, Diego P. Pinto-Roa, Jose Luis Vazquez Noguera

path = extractBefore(impath, "input");
image = imread(impath);
if ~ismatrix(image)
    image = rgb2gray(image);
end
[m,n] = size(image);
lower_limit1=2; % lower-limit rx
lower_limit2=2; % lower-limit ry
lower_limit3=0; % lower-limit c 
upper_limit1=round(m/2);
upper_limit2=round(n/2);
upper_limit3=0.5;

omega = 100; % cantidad de particulas

pareto=[];
pname=[];
c1=2;
c2=2;
delta=[double((upper_limit1-lower_limit1)/2),double((upper_limit2-lower_limit2)/2),double((upper_limit3-lower_limit3)/2)];

% ----Generate the swarm-----
rng('shuffle');
x=[randi([lower_limit1,upper_limit1],omega,1), randi([lower_limit2,upper_limit2],omega,1), lower_limit3+(upper_limit3-lower_limit3)*rand(omega,1)];
xp=x;
v=[randi([-(upper_limit1-lower_limit1),upper_limit1-lower_limit1],omega,1), randi([-(upper_limit2-lower_limit2),upper_limit2-lower_limit2],omega,1), -(upper_limit3-lower_limit3)+2*(upper_limit3-lower_limit3)*rand(omega,1)];
fitnessXP=zeros(omega,2);
for i=1:omega
    K=adapthisteq(image,'NumTiles',[x(i,1) x(i,2)],'ClipLimit',x(i,3));
    e=entropy(K)/(log2(256));
    ss=ssim(K,image);
    fitnessXP(i,:)=[ss,e];
end

for t=1:tmax
    rng('shuffle');
    z=rand;
    w=0.5*(tmax-t)/tmax+0.4*4*z*(1-z);
    for i=1:omega
        disp(['t=,',num2str(t),' i=',num2str(i)]);
        
        %Cálculo de velocidad para la i-ésima partícula
        rng('shuffle');
        r1=rand;
        r2=rand;
        v(i,:)=w*v(i,:)+c1*r1*(xp(i,:)-x(i,:));
        for j=1:3
            if v(i,j)>delta(j)
                v(i,j)=delta(j);
            elseif v(i,j)<-delta(j)
                v(i,j)=-delta(j);
            end
        end
        
        %Nueva posición de la partícula
        x(i,:)=x(i,:)+v(i,:);
        if x(i,1)<lower_limit1 || x(i,1)>upper_limit1
            x(i,1)=randi([lower_limit1,upper_limit1]);
        end
        if x(i,2)<lower_limit2 || x(i,2)>upper_limit2
            x(i,2)=randi([lower_limit2,upper_limit2]);
        end
        if x(i,3)<lower_limit3 || x(i,3)>upper_limit3
            x(i,3)=lower_limit3+(upper_limit3-lower_limit3)*rand;
        end

        %CLAHE
        K=adapthisteq(image,'NumTiles',[round(x(i,1)) round(x(i,2))],'ClipLimit',x(i,3));
        e=entropy(K)/(log2(256));
        ss=ssim(K,image);
        fitnessX=[ss,e];
        if((fitnessX(1)>=fitnessXP(i,1) && fitnessX(2)>=fitnessXP(i,2)) && (fitnessX(1)>fitnessXP(i,1) || fitnessX(2)>fitnessXP(i,2)))
            xp(i,:)=x(i,:);
            fitnessXP(i,:)=fitnessX;
            pareto=[pareto;xp(i,:),fitnessXP(i,:)];
            name=strcat('CLAHE-I',num2str(t),'-M',num2str(round(xp(i,1))),'-N',num2str(round(xp(i,2))),'-C',num2str(xp(i,3)),'.png');
            pname=[pname;convertCharsToStrings(name)];
            imwrite(K, strcat(path,'output/',name));
        end
    end
end