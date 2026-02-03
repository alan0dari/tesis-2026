function [particulas, pareto_final, names_final] = mopso_clahe(impath,tmax)
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
upper_limit3=1;

omega = 100; % cantidad de particulas

pareto=[];
pname=[];
c1=2;
c2=2;
delta=[double((upper_limit1-lower_limit1)/2),double((upper_limit2-lower_limit2)/2),double((upper_limit3-lower_limit3)/2)];

% ----Generate the swarm-----
rng('shuffle');
x=[randi([lower_limit1,upper_limit1],omega,1), randi([lower_limit2,upper_limit2],omega,1), lower_limit3+(upper_limit3-lower_limit3)*rand(omega,1)];
v=[randi([-(upper_limit1-lower_limit1),upper_limit1-lower_limit1],omega,1), randi([-(upper_limit2-lower_limit2),upper_limit2-lower_limit2],omega,1), -(upper_limit3-lower_limit3)+2*(upper_limit3-lower_limit3)*rand(omega,1)];
xp=x;
fitnessXG=[0,0];
fitnessXP=zeros(omega,2);
for i=1:omega
    disp(['t=0, i=',num2str(i)]);
    K=adapthisteq(image,'NumTiles',[x(i,1) x(i,2)],'ClipLimit',x(i,3));
    ss=ssim(K,image);
    e=entropy(K);
    fitnessXP(i,:)=[e, ss];
    if(((fitnessXP(i,1)>=fitnessXG(1) && fitnessXP(i,2)>=fitnessXG(2))) || (fitnessXP(i,1)>fitnessXG(1) || fitnessXP(i,2)>fitnessXG(2)))
        xg=x(i,:);
        fitnessXG=fitnessXP(i,:);
     end
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
        v(i,:)=w*v(i,:)+c1*r1*(xp(i,:)-x(i,:))+c2*r2*(xg-x(i,:));
        for j=1:3
            if v(i,j)>delta(j)
                v(i,j)=delta(j);
            elseif v(i,j)<-delta(j)
                v(i,j)=-delta(j);
            end
        end
        
        %Nueva posición de la partícula
        x(i,:)=x(i,:)+v(i,:);
        if x(i,1)<lower_limit1
            x(i,1)=lower_limit1;
        elseif x(i,1)>upper_limit1
            x(i,1)=upper_limit1;
        end
        if x(i,2)<lower_limit2
            x(i,2)=lower_limit2;
        elseif x(i,2)>upper_limit2
            x(i,2)=upper_limit2;
        end
        if x(i,3)<lower_limit3
            x(i,3)=lower_limit3;
        elseif x(i,3)>upper_limit3
            x(i,3)=upper_limit3;
        end

        %CLAHE
        K=adapthisteq(image,'NumTiles',[round(x(i,1)) round(x(i,2))],'ClipLimit',x(i,3));
        ss=ssim(K,image);
        e=entropy(K);
        fitnessX=[e, ss];
        if(((fitnessX(1)>=fitnessXP(i,1) && fitnessX(2)>=fitnessXP(i,2))) || (fitnessX(1)>fitnessXP(i,1) || fitnessX(2)>fitnessXP(i,2)))
            xp(i,:)=x(i,:);
            fitnessXP(i,:)=fitnessX;
        end
        if(((fitnessX(1)>=fitnessXG(1) && fitnessX(2)>=fitnessXG(2))) || (fitnessX(1)>fitnessXG(1) || fitnessX(2)>fitnessXG(2)))
            xg=x(i,:);
            fitnessXG=fitnessX;
            pareto=[pareto;xg,fitnessXG];
            name=strcat('CLAHE__i',num2str(t),'__m',num2str(round(xg(1))),'__n',num2str(round(xg(2))),'__cl',num2str(xg(3)),'.png');
            pname=[pname;convertCharsToStrings(name)];
        end
    end
end

pareto(:,1:2)=round(pareto(:,1:2));
particulas=pareto;
membership=find_pareto_frontier(pareto(:,4:5),[1 1]);

pareto_final=[];
names_final=[];
for i=1:length(membership)
    if membership(i)
        pareto_final=[pareto_final;pareto(i,:)];
        names_final=[names_final;pname(i)];
%         K=adapthisteq(image,'NumTiles',[pareto(i,1) pareto(i,2)],'ClipLimit',pareto(i,3));
%         filename=strcat(path,'output/pareto/',convertStringsToChars(pname(i)));
%         imwrite(K, filename);
    end
end