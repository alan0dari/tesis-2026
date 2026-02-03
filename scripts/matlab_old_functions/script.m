function [] = script(type, testnum, imtype, tmax) %Ej: script("Teeth", "02", "jpg", 100)
impath = strcat("Dataset - ", type, "/Input/Test ", testnum, "/input.", imtype);

MM=[1 1];
[Particulas, Pareto, Names]=mopso_clahe(impath, tmax);
path = extractBefore(impath, "Input");
image = imread(impath);
if ~ismatrix(image)
    image=rgb2gray(image);
end

DM = Pareto(:,4:5);
[m,n] = size(DM);
[m2,n2] = size(Particulas);
w1 = weight_entropy(m,n,DM);
w2 = [0.6932, 0.3068];
w3 = [0.90, 0.10];

[~, indice_smarter1] = smarter(m,n,DM,w1,MM);
[~, indice_fuzzy1] = fuzzy(m,n,DM,w1,MM);
[~, indice_topsis1] = topsis(m,n,DM,w1,MM);
[~, indice_gra1] = Fun_GRAt(m,n,DM,w1,MM);
[~, indice_codas1] = Fun_CODAS(m,n,DM,w1,MM);
[~, indice_mabac1] = Fun_MABAC(m,n,DM,w1,MM);
[~, indice_vikor1] = Fun_VIKOR(m,n,DM,w1,MM);
[~, indice_promethee1] = Fun_PROMETHEE(m,n,DM,w1,MM);

[~, indice_smarter2] = smarter(m,n,DM,w2,MM);
[~, indice_fuzzy2] = fuzzy(m,n,DM,w2,MM);
[~, indice_topsis2] = topsis(m,n,DM,w2,MM);
[~, indice_gra2] = Fun_GRAt(m,n,DM,w2,MM);
[~, indice_codas2] = Fun_CODAS(m,n,DM,w2,MM);
[~, indice_mabac2] = Fun_MABAC(m,n,DM,w2,MM);
[~, indice_vikor2] = Fun_VIKOR(m,n,DM,w2,MM);
[~, indice_promethee2] = Fun_PROMETHEE(m,n,DM,w2,MM);

[~, indice_smarter3]=smarter(m,n,DM,w3,MM);
[~, indice_fuzzy3]=fuzzy(m,n,DM,w3,MM);
[~, indice_topsis3]=topsis(m,n,DM,w3,MM);
[~, indice_gra3]=Fun_GRAt(m,n,DM,w3,MM);
[~, indice_codas3]=Fun_CODAS(m,n,DM,w3,MM);
[~, indice_mabac3]=Fun_MABAC(m,n,DM,w3,MM);
[~, indice_vikor3]=Fun_VIKOR(m,n,DM,w3,MM);
[~, indice_promethee3]=Fun_PROMETHEE(m,n,DM,w3,MM);


K=adapthisteq(image,'NumTiles',[Pareto(indice_smarter1,1) Pareto(indice_smarter1,2)],'ClipLimit',Pareto(indice_smarter1,3));
filename=strcat(path, "Output - SMARTER/Test ", testnum, "/output__smarter__w1__", convertStringsToChars(Names(indice_smarter1)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_fuzzy1,1) Pareto(indice_fuzzy1,2)],'ClipLimit',Pareto(indice_fuzzy1,3));
filename=strcat(path, "Output - Bellman-Zadeh/Test ", testnum, "/output__fuzzy__w1__", convertStringsToChars(Names(indice_fuzzy1)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_topsis1,1) Pareto(indice_topsis1,2)],'ClipLimit',Pareto(indice_topsis1,3));
filename=strcat(path, "Output - TOPSIS/Test ", testnum, "/output__topsis__w1__", convertStringsToChars(Names(indice_topsis1)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_gra1,1) Pareto(indice_gra1,2)],'ClipLimit',Pareto(indice_gra1,3));
filename=strcat(path, "Output - GRA/Test ", testnum, "/output__gra__w1__", convertStringsToChars(Names(indice_gra1)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_codas1,1) Pareto(indice_codas1,2)],'ClipLimit',Pareto(indice_codas1,3));
filename=strcat(path, "Output - CODAS/Test ", testnum, "/output__codas__w1__", convertStringsToChars(Names(indice_codas1)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_mabac1,1) Pareto(indice_mabac1,2)],'ClipLimit',Pareto(indice_mabac1,3));
filename=strcat(path, "Output - MABAC/Test ", testnum, "/output__mabac__w1__", convertStringsToChars(Names(indice_mabac1)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_vikor1,1) Pareto(indice_vikor1,2)],'ClipLimit',Pareto(indice_vikor1,3));
filename=strcat(path, "Output - VIKOR/Test ", testnum, "/output__vikor__w1__", convertStringsToChars(Names(indice_vikor1)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_promethee1,1) Pareto(indice_promethee1,2)],'ClipLimit',Pareto(indice_promethee1,3));
filename=strcat(path, "Output - PROMETHEE II/Test ", testnum, "/output__promethee__w1__", convertStringsToChars(Names(indice_promethee1)));
imwrite(K, filename);


K=adapthisteq(image,'NumTiles',[Pareto(indice_smarter2,1) Pareto(indice_smarter2,2)],'ClipLimit',Pareto(indice_smarter2,3));
filename=strcat(path, "Output - SMARTER/Test ", testnum, "/output__smarter__w2__", convertStringsToChars(Names(indice_smarter2)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_fuzzy2,1) Pareto(indice_fuzzy2,2)],'ClipLimit',Pareto(indice_fuzzy2,3));
filename=strcat(path, "Output - Bellman-Zadeh/Test ", testnum, "/output__fuzzy__w2__", convertStringsToChars(Names(indice_fuzzy2)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_topsis2,1) Pareto(indice_topsis2,2)],'ClipLimit',Pareto(indice_topsis2,3));
filename=strcat(path, "Output - TOPSIS/Test ", testnum, "/output__topsis__w2__", convertStringsToChars(Names(indice_topsis2)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_gra2,1) Pareto(indice_gra2,2)],'ClipLimit',Pareto(indice_gra2,3));
filename=strcat(path, "Output - GRA/Test ", testnum, "/output__gra__w2__", convertStringsToChars(Names(indice_gra2)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_codas2,1) Pareto(indice_codas2,2)],'ClipLimit',Pareto(indice_codas2,3));
filename=strcat(path, "Output - CODAS/Test ", testnum, "/output__codas__w2__", convertStringsToChars(Names(indice_codas2)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_mabac2,1) Pareto(indice_mabac2,2)],'ClipLimit',Pareto(indice_mabac2,3));
filename=strcat(path, "Output - MABAC/Test ", testnum, "/output__mabac__w2__", convertStringsToChars(Names(indice_mabac2)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_vikor2,1) Pareto(indice_vikor2,2)],'ClipLimit',Pareto(indice_vikor2,3));
filename=strcat(path, "Output - VIKOR/Test ", testnum, "/output__vikor__w2__", convertStringsToChars(Names(indice_vikor2)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_promethee2,1) Pareto(indice_promethee2,2)],'ClipLimit',Pareto(indice_promethee2,3));
filename=strcat(path, "Output - PROMETHEE II/Test ", testnum, "/output__promethee__w2__", convertStringsToChars(Names(indice_promethee2)));
imwrite(K, filename);


K=adapthisteq(image,'NumTiles',[Pareto(indice_smarter3,1) Pareto(indice_smarter3,2)],'ClipLimit',Pareto(indice_smarter3,3));
filename=strcat(path, "Output - SMARTER/Test ", testnum, "/output__smarter__w3__", convertStringsToChars(Names(indice_smarter3)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_fuzzy3,1) Pareto(indice_fuzzy3,2)],'ClipLimit',Pareto(indice_fuzzy3,3));
filename=strcat(path, "Output - Bellman-Zadeh/Test ", testnum, "/output__fuzzy__w3__", convertStringsToChars(Names(indice_fuzzy3)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_topsis3,1) Pareto(indice_topsis3,2)],'ClipLimit',Pareto(indice_topsis3,3));
filename=strcat(path, "Output - TOPSIS/Test ", testnum, "/output__topsis__w3__", convertStringsToChars(Names(indice_topsis3)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_gra3,1) Pareto(indice_gra3,2)],'ClipLimit',Pareto(indice_gra3,3));
filename=strcat(path, "Output - GRA/Test ", testnum, "/output__gra__w3__", convertStringsToChars(Names(indice_gra3)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_codas3,1) Pareto(indice_codas3,2)],'ClipLimit',Pareto(indice_codas3,3));
filename=strcat(path, "Output - CODAS/Test ", testnum, "/output__codas__w3__", convertStringsToChars(Names(indice_codas3)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_mabac3,1) Pareto(indice_mabac3,2)],'ClipLimit',Pareto(indice_mabac3,3));
filename=strcat(path, "Output - MABAC/Test ", testnum, "/output__mabac__w3__", convertStringsToChars(Names(indice_mabac3)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_vikor3,1) Pareto(indice_vikor3,2)],'ClipLimit',Pareto(indice_vikor3,3));
filename=strcat(path, "Output - VIKOR/Test ", testnum, "/output__vikor__w3__", convertStringsToChars(Names(indice_vikor2)));
imwrite(K, filename);
K=adapthisteq(image,'NumTiles',[Pareto(indice_promethee3,1) Pareto(indice_promethee3,2)],'ClipLimit',Pareto(indice_promethee3,3));
filename=strcat(path, "Output - PROMETHEE II/Test ", testnum, "/output__promethee__w3__", convertStringsToChars(Names(indice_promethee3)));
imwrite(K, filename);


display(strcat("Método: SMARTER, w=[",num2str(w1),"], m=", num2str(Pareto(indice_smarter1,1)), " n=", num2str(Pareto(indice_smarter1,2))," c=", num2str(Pareto(indice_smarter1,3)), " e=", num2str(Pareto(indice_smarter1,4)),  " SSIM=", num2str(Pareto(indice_smarter1,5))));
display(strcat("Método: Fuzzy, w=[",num2str(w1),"], m=", num2str(Pareto(indice_fuzzy1,1)), " n=", num2str(Pareto(indice_fuzzy1,2))," c=", num2str(Pareto(indice_fuzzy1,3)), " e=", num2str(Pareto(indice_fuzzy1,4)),  " SSIM=", num2str(Pareto(indice_fuzzy1,5))));
display(strcat("Método: TOPSIS, w=[",num2str(w1),"], m=", num2str(Pareto(indice_topsis1,1)), " n=", num2str(Pareto(indice_topsis1,2))," c=", num2str(Pareto(indice_topsis1,3)), " e=", num2str(Pareto(indice_topsis1,4)),  " SSIM=", num2str(Pareto(indice_topsis1,5))));
display(strcat("Método: GRA, w=[",num2str(w1),"], m=", num2str(Pareto(indice_gra1,1)), " n=", num2str(Pareto(indice_gra1,2))," c=", num2str(Pareto(indice_gra1,3)), " e=", num2str(Pareto(indice_gra1,4)),  " SSIM=", num2str(Pareto(indice_gra1,5))));
display(strcat("Método: CODAS, w=[",num2str(w1),"], m=", num2str(Pareto(indice_codas1,1)), " n=", num2str(Pareto(indice_codas1,2))," c=", num2str(Pareto(indice_codas1,3)), " e=", num2str(Pareto(indice_codas1,4)),  " SSIM=", num2str(Pareto(indice_codas1,5))));
display(strcat("Método: MABAC, w=[",num2str(w1),"], m=", num2str(Pareto(indice_mabac1,1)), " n=", num2str(Pareto(indice_mabac1,2))," c=", num2str(Pareto(indice_mabac1,3)), " e=", num2str(Pareto(indice_mabac1,4)),  " SSIM=", num2str(Pareto(indice_mabac1,5))));
display(strcat("Método: VIKOR, w=[",num2str(w1),"], m=", num2str(Pareto(indice_vikor1,1)), " n=", num2str(Pareto(indice_vikor1,2))," c=", num2str(Pareto(indice_vikor1,3)), " e=", num2str(Pareto(indice_vikor1,4)),  " SSIM=", num2str(Pareto(indice_vikor1,5))));
display(strcat("Método: PROMETHEE II, w=[",num2str(w1),"], m=", num2str(Pareto(indice_promethee1,1)), " n=", num2str(Pareto(indice_promethee1,2))," c=", num2str(Pareto(indice_promethee1,3)), " e=", num2str(Pareto(indice_promethee1,4)),  " SSIM=", num2str(Pareto(indice_promethee1,5))));


display(strcat("Método: SMARTER, w=[",num2str(w2),"], m=", num2str(Pareto(indice_smarter2,1)), " n=", num2str(Pareto(indice_smarter2,2))," c=", num2str(Pareto(indice_smarter2,3)), " e=", num2str(Pareto(indice_smarter2,4)),  " SSIM=", num2str(Pareto(indice_smarter2,5))));
display(strcat("Método: Fuzzy, w=[",num2str(w2),"], m=", num2str(Pareto(indice_fuzzy2,1)), " n=", num2str(Pareto(indice_fuzzy2,2))," c=", num2str(Pareto(indice_fuzzy2,3)), " e=", num2str(Pareto(indice_fuzzy2,4)),  " SSIM=", num2str(Pareto(indice_fuzzy2,5))));
display(strcat("Método: TOPSIS, w=[",num2str(w2),"], m=", num2str(Pareto(indice_topsis2,1)), " n=", num2str(Pareto(indice_topsis2,2))," c=", num2str(Pareto(indice_topsis2,3)), " e=", num2str(Pareto(indice_topsis2,4)),  " SSIM=", num2str(Pareto(indice_topsis2,5))));
display(strcat("Método: GRA, w=[",num2str(w2),"], m=", num2str(Pareto(indice_gra2,1)), " n=", num2str(Pareto(indice_gra2,2))," c=", num2str(Pareto(indice_gra2,3)), " e=", num2str(Pareto(indice_gra2,4)),  " SSIM=", num2str(Pareto(indice_gra2,5))));
display(strcat("Método: CODAS, w=[",num2str(w2),"], m=", num2str(Pareto(indice_codas2,1)), " n=", num2str(Pareto(indice_codas2,2))," c=", num2str(Pareto(indice_codas2,3)), " e=", num2str(Pareto(indice_codas2,4)),  " SSIM=", num2str(Pareto(indice_codas2,5))));
display(strcat("Método: MABAC, w=[",num2str(w2),"], m=", num2str(Pareto(indice_mabac2,1)), " n=", num2str(Pareto(indice_mabac2,2))," c=", num2str(Pareto(indice_mabac2,3)), " e=", num2str(Pareto(indice_mabac2,4)),  " SSIM=", num2str(Pareto(indice_mabac2,5))));
display(strcat("Método: VIKOR, w=[",num2str(w2),"], m=", num2str(Pareto(indice_vikor2,1)), " n=", num2str(Pareto(indice_vikor2,2))," c=", num2str(Pareto(indice_vikor2,3)), " e=", num2str(Pareto(indice_vikor2,4)),  " SSIM=", num2str(Pareto(indice_vikor2,5))));
display(strcat("Método: PROMETHEE II, w=[",num2str(w2),"], m=", num2str(Pareto(indice_promethee2,1)), " n=", num2str(Pareto(indice_promethee2,2))," c=", num2str(Pareto(indice_promethee2,3)), " e=", num2str(Pareto(indice_promethee2,4)),  " SSIM=", num2str(Pareto(indice_promethee2,5))));


display(strcat("Método: SMARTER, w=[",num2str(w3),"], m=", num2str(Pareto(indice_smarter3,1)), " n=", num2str(Pareto(indice_smarter3,2))," c=", num2str(Pareto(indice_smarter3,3)), " e=", num2str(Pareto(indice_smarter3,4)),  " SSIM=", num2str(Pareto(indice_smarter3,5))));
display(strcat("Método: Fuzzy, w=[",num2str(w3),"], m=", num2str(Pareto(indice_fuzzy3,1)), " n=", num2str(Pareto(indice_fuzzy3,2))," c=", num2str(Pareto(indice_fuzzy3,3)), " e=", num2str(Pareto(indice_fuzzy3,4)),  " SSIM=", num2str(Pareto(indice_fuzzy3,5))));
display(strcat("Método: TOPSIS, w=[",num2str(w3),"], m=", num2str(Pareto(indice_topsis3,1)), " n=", num2str(Pareto(indice_topsis3,2))," c=", num2str(Pareto(indice_topsis3,3)), " e=", num2str(Pareto(indice_topsis3,4)),  " SSIM=", num2str(Pareto(indice_topsis3,5))));
display(strcat("Método: GRA, w=[",num2str(w3),"], m=", num2str(Pareto(indice_gra3,1)), " n=", num2str(Pareto(indice_gra3,2))," c=", num2str(Pareto(indice_gra3,3)), " e=", num2str(Pareto(indice_gra3,4)),  " SSIM=", num2str(Pareto(indice_gra3,5))));
display(strcat("Método: CODAS, w=[",num2str(w3),"], m=", num2str(Pareto(indice_codas3,1)), " n=", num2str(Pareto(indice_codas3,2))," c=", num2str(Pareto(indice_codas3,3)), " e=", num2str(Pareto(indice_codas3,4)),  " SSIM=", num2str(Pareto(indice_codas3,5))));
display(strcat("Método: MABAC, w=[",num2str(w3),"], m=", num2str(Pareto(indice_mabac3,1)), " n=", num2str(Pareto(indice_mabac3,2))," c=", num2str(Pareto(indice_mabac3,3)), " e=", num2str(Pareto(indice_mabac3,4)),  " SSIM=", num2str(Pareto(indice_mabac3,5))));
display(strcat("Método: VIKOR, w=[",num2str(w3),"], m=", num2str(Pareto(indice_vikor3,1)), " n=", num2str(Pareto(indice_vikor3,2))," c=", num2str(Pareto(indice_vikor3,3)), " e=", num2str(Pareto(indice_vikor3,4)),  " SSIM=", num2str(Pareto(indice_vikor3,5))));
display(strcat("Método: PROMETHEE II, w=[",num2str(w3),"], m=", num2str(Pareto(indice_promethee3,1)), " n=", num2str(Pareto(indice_promethee3,2))," c=", num2str(Pareto(indice_promethee3,3)), " e=", num2str(Pareto(indice_promethee3,4)),  " SSIM=", num2str(Pareto(indice_promethee3,5))));


fig_pareto = figure('WindowState','maximized','Name','Frente Pareto','NumberTitle','off');
axes1 = axes;
hold(axes1,'on');
scatter(Particulas(:,4), Particulas(:,5));
hold on;
scatter(Pareto(:,4),Pareto(:,5),25,'MarkerEdgeColor','r','MarkerFaceColor','r','LineWidth',1.5);
xlabel('Entropía','FontWeight','bold','FontSize',24,'FontName','Lucida Bright');
ylabel('SSIM','FontWeight','bold','FontSize',24,'FontName','Lucida Bright');
legend({'Partículas', 'Frente Pareto'});
set(legend,'Location','southwest','FontAngle','italic','FontSize',14,'FontName','Lucida Bright');
set(axes1,'FontName','Lucida Bright','FontSize',18);
annotation(fig_pareto,'textbox',...
    [0.157662032114814 0.534562144131562 0.278518510438778 0.104454682444647],...
    'String',{strcat('Total partículas: ', string(m2)), strcat('Frente Pareto :',string(m),' soluciones')},...
    'FontSize',18,...
    'FontName','Lucida Bright',...
    'FontAngle','italic',...
    'FitBoxToText','off');
filename = strcat(path, "Data/Test ", testnum, "/pareto.svg");
filename2 = strcat(path, "Data/Test ", testnum, "/pareto.fig");
saveas(fig_pareto, filename);
saveas(fig_pareto, filename2);

fig_decision1 = figure('WindowState','maximized','Name','Métodos de decisión: w1','NumberTitle','off');
axes1 = axes;
hold(axes1,'on');
scatter(Pareto(:,4),Pareto(:,5));
hold on
scatter(DM(indice_smarter1,1),DM(indice_smarter1,2),'s','MarkerEdgeColor','r','MarkerFaceColor','r','LineWidth',1.5);
scatter(DM(indice_fuzzy1,1),DM(indice_fuzzy1,2),'d','MarkerEdgeColor','g','MarkerFaceColor','g','LineWidth',1.5);
scatter(DM(indice_topsis1,1),DM(indice_topsis1,2),'^','MarkerEdgeColor','b','MarkerFaceColor','b','LineWidth',1.5);
scatter(DM(indice_gra1,1),DM(indice_gra1,2),'p','MarkerEdgeColor','y','MarkerFaceColor','y','LineWidth',1.5);
scatter(DM(indice_codas1,1),DM(indice_codas1,2),'h','MarkerEdgeColor','m','MarkerFaceColor','m','LineWidth',1.5);
scatter(DM(indice_mabac1,1),DM(indice_mabac1,2),'x','MarkerEdgeColor','c','MarkerFaceColor','c','LineWidth',1.5);
scatter(DM(indice_vikor1,1),DM(indice_vikor1,2),'*','MarkerEdgeColor','k','MarkerFaceColor','k','LineWidth',1.5);
scatter(DM(indice_promethee1,1),DM(indice_promethee1,2),'+','MarkerEdgeColor','r','MarkerFaceColor','b','LineWidth',1.5);
xlabel('Entropía','FontWeight','bold','FontSize',24,'FontName','Lucida Bright');
ylabel('SSIM','FontWeight','bold','FontSize',24,'FontName','Lucida Bright');
legend({'Frente Pareto', 'Selección SMARTER', 'Selección Fuzzy', 'Selección TOPSIS', 'Selección GRA', 'Selección CODAS', 'Selección MABAC', 'Selección VIKOR', 'Selección PROMETHEE II'});
set(legend,'Location','southwest','FontAngle','italic','FontSize',14,'FontName','Lucida Bright');
set(axes1,'FontName','Lucida Bright','FontSize',18);
annotation(fig_decision1,'textbox',...
    [0.157662032114814 0.579108993802426 0.278518510438778 0.0599078327737829],...
    'String',{strcat('Frente Pareto: ', string(m),' soluciones')},...
    'FontSize',18,...
    'FontName','Lucida Bright',...
    'FontAngle','italic',...
    'FitBoxToText','off');
filename = strcat(path, "Data/Test ", testnum, "/selecciones_metodos_w1.svg");
filename2 = strcat(path, "Data/Test ", testnum, "/selecciones_metodos_w1.fig");
saveas(fig_decision1,filename);
saveas(fig_decision1,filename2);


fig_decision2 = figure('WindowState','maximized','Name','Métodos de decisión: w2','NumberTitle','off');
axes1 = axes;
hold(axes1,'on');
scatter(Pareto(:,4),Pareto(:,5));
hold on
scatter(DM(indice_smarter2,1),DM(indice_smarter2,2),'s','MarkerEdgeColor','r','MarkerFaceColor','r','LineWidth',1.5);
scatter(DM(indice_fuzzy2,1),DM(indice_fuzzy2,2),'d','MarkerEdgeColor','g','MarkerFaceColor','g','LineWidth',1.5);
scatter(DM(indice_topsis2,1),DM(indice_topsis2,2),'^','MarkerEdgeColor','b','MarkerFaceColor','b','LineWidth',1.5);
scatter(DM(indice_gra2,1),DM(indice_gra2,2),'p','MarkerEdgeColor','y','MarkerFaceColor','y','LineWidth',1.5);
scatter(DM(indice_codas2,1),DM(indice_codas2,2),'h','MarkerEdgeColor','m','MarkerFaceColor','m','LineWidth',1.5);
scatter(DM(indice_mabac2,1),DM(indice_mabac2,2),'x','MarkerEdgeColor','c','MarkerFaceColor','c','LineWidth',1.5);
scatter(DM(indice_vikor2,1),DM(indice_vikor2,2),'*','MarkerEdgeColor','k','MarkerFaceColor','k','LineWidth',1.5);
scatter(DM(indice_promethee2,1),DM(indice_promethee2,2),'+','MarkerEdgeColor','r','MarkerFaceColor','b','LineWidth',1.5);
xlabel('Entropía','FontWeight','bold','FontSize',24,'FontName','Lucida Bright');
ylabel('SSIM','FontWeight','bold','FontSize',24,'FontName','Lucida Bright');
legend({'Frente Pareto', 'Selección SMARTER', 'Selección Fuzzy', 'Selección TOPSIS', 'Selección GRA', 'Selección CODAS', 'Selección MABAC', 'Selección VIKOR', 'Selección PROMETHEE II'});
set(legend,'Location','southwest','FontAngle','italic','FontSize',14,'FontName','Lucida Bright');
set(axes1,'FontName','Lucida Bright','FontSize',18);
annotation(fig_decision2,'textbox',...
    [0.157662032114814 0.579108993802426 0.278518510438778 0.0599078327737829],...
    'String',{strcat('Frente Pareto: ', string(m),' soluciones')},...
    'FontSize',18,...
    'FontName','Lucida Bright',...
    'FontAngle','italic',...
    'FitBoxToText','off');
filename = strcat(path, "Data/Test ", testnum, "/selecciones_metodos_w2.svg");
filename2 = strcat(path, "Data/Test ", testnum, "/selecciones_metodos_w2.fig");
saveas(fig_decision2,filename);
saveas(fig_decision2,filename2);


fig_decision3 = figure('WindowState','maximized','Name','Métodos de decisión: w3','NumberTitle','off');
axes1 = axes;
hold(axes1,'on');
scatter(Pareto(:,4),Pareto(:,5));
hold on
scatter(DM(indice_smarter3,1),DM(indice_smarter3,2),'s','MarkerEdgeColor','r','MarkerFaceColor','r','LineWidth',1.5);
scatter(DM(indice_fuzzy3,1),DM(indice_fuzzy3,2),'d','MarkerEdgeColor','g','MarkerFaceColor','g','LineWidth',1.5);
scatter(DM(indice_topsis3,1),DM(indice_topsis3,2),'^','MarkerEdgeColor','b','MarkerFaceColor','b','LineWidth',1.5);
scatter(DM(indice_gra3,1),DM(indice_gra3,2),'p','MarkerEdgeColor','y','MarkerFaceColor','y','LineWidth',1.5);
scatter(DM(indice_codas3,1),DM(indice_codas3,2),'h','MarkerEdgeColor','m','MarkerFaceColor','m','LineWidth',1.5);
scatter(DM(indice_mabac3,1),DM(indice_mabac3,2),'x','MarkerEdgeColor','c','MarkerFaceColor','c','LineWidth',1.5);
scatter(DM(indice_vikor3,1),DM(indice_vikor3,2),'*','MarkerEdgeColor','k','MarkerFaceColor','k','LineWidth',1.5);
scatter(DM(indice_promethee3,1),DM(indice_promethee3,2),'+','MarkerEdgeColor','r','MarkerFaceColor','b','LineWidth',1.5);
xlabel('Entropía','FontWeight','bold','FontSize',24,'FontName','Lucida Bright');
ylabel('SSIM','FontWeight','bold','FontSize',24,'FontName','Lucida Bright');
legend({'Frente Pareto', 'Selección SMARTER', 'Selección Fuzzy', 'Selección TOPSIS', 'Selección GRA', 'Selección CODAS', 'Selección MABAC', 'Selección VIKOR', 'Selección PROMETHEE II'});
set(legend,'Location','southwest','FontAngle','italic','FontSize',14,'FontName','Lucida Bright');
set(axes1,'FontName','Lucida Bright','FontSize',18);
annotation(fig_decision3,'textbox',...
    [0.157662032114814 0.579108993802426 0.278518510438778 0.0599078327737829],...
    'String',{strcat('Frente Pareto: ', string(m),' soluciones')},...
    'FontSize',18,...
    'FontName','Lucida Bright',...
    'FontAngle','italic',...
    'FitBoxToText','off');
filename = strcat(path, "Data/Test ", testnum, "/selecciones_metodos_w3.svg");
filename2 = strcat(path, "Data/Test ", testnum, "/selecciones_metodos_w3.fig");
saveas(fig_decision3,filename);
saveas(fig_decision3,filename2);

T = table(Names, Pareto(:,1), Pareto(:,2), Pareto(:,3), Pareto(:,4), Pareto(:,5),'VariableNames',{'Nombre','Rx','Ry','CL','Entropia','SSIM'});
writetable(T, strcat(path, "Data/Test ", testnum, "/pareto.xlsx"), 'Sheet',1,'Range','A1:H10000');


T = table({'SMARTER';'Fuzzy';'TOPSIS';'GRA';'CODAS';'MABAC';'VIKOR';'PROMETHEE II'},...
[Pareto(indice_smarter1,1);Pareto(indice_fuzzy1,1);Pareto(indice_topsis1,1);Pareto(indice_gra1,1);Pareto(indice_codas1,1);Pareto(indice_mabac1,1);Pareto(indice_vikor1,1);Pareto(indice_promethee1,1)],...
[Pareto(indice_smarter1,2);Pareto(indice_fuzzy1,2);Pareto(indice_topsis1,2);Pareto(indice_gra1,2);Pareto(indice_codas1,2);Pareto(indice_mabac1,2);Pareto(indice_vikor1,2);Pareto(indice_promethee1,2)],...
[Pareto(indice_smarter1,3);Pareto(indice_fuzzy1,3);Pareto(indice_topsis1,3);Pareto(indice_gra1,3);Pareto(indice_codas1,3);Pareto(indice_mabac1,3);Pareto(indice_vikor1,3);Pareto(indice_promethee1,3)],...
[Pareto(indice_smarter1,4);Pareto(indice_fuzzy1,4);Pareto(indice_topsis1,4);Pareto(indice_gra1,4);Pareto(indice_codas1,4);Pareto(indice_mabac1,4);Pareto(indice_vikor1,4);Pareto(indice_promethee1,4)],...
[Pareto(indice_smarter1,5);Pareto(indice_fuzzy1,5);Pareto(indice_topsis1,5);Pareto(indice_gra1,5);Pareto(indice_codas1,5);Pareto(indice_mabac1,5);Pareto(indice_vikor1,5);Pareto(indice_promethee1,5)],...
'VariableNames',{'Metodo','Rx','Ry','CL','Entropia','SSIM'},'RowNames',{'SMARTER','Fuzzy','TOPSIS','GRA','CODAS','MABAC','VIKOR','PROMETHEE II'});
writetable(T, strcat(path, "Data/Test ", testnum, "/decisiones_w1.xlsx"), 'Sheet',1,'Range','A1:H10000');


T = table({'SMARTER';'Fuzzy';'TOPSIS';'GRA';'CODAS';'MABAC';'VIKOR';'PROMETHEE II'},...
[Pareto(indice_smarter2,1);Pareto(indice_fuzzy2,1);Pareto(indice_topsis2,1);Pareto(indice_gra2,1);Pareto(indice_codas2,1);Pareto(indice_mabac2,1);Pareto(indice_vikor2,1);Pareto(indice_promethee2,1)],...
[Pareto(indice_smarter2,2);Pareto(indice_fuzzy2,2);Pareto(indice_topsis2,2);Pareto(indice_gra2,2);Pareto(indice_codas2,2);Pareto(indice_mabac2,2);Pareto(indice_vikor2,2);Pareto(indice_promethee2,2)],...
[Pareto(indice_smarter2,3);Pareto(indice_fuzzy2,3);Pareto(indice_topsis2,3);Pareto(indice_gra2,3);Pareto(indice_codas2,3);Pareto(indice_mabac2,3);Pareto(indice_vikor2,3);Pareto(indice_promethee2,3)],...
[Pareto(indice_smarter2,4);Pareto(indice_fuzzy2,4);Pareto(indice_topsis2,4);Pareto(indice_gra2,4);Pareto(indice_codas2,4);Pareto(indice_mabac2,4);Pareto(indice_vikor2,4);Pareto(indice_promethee2,4)],...
[Pareto(indice_smarter2,5);Pareto(indice_fuzzy2,5);Pareto(indice_topsis2,5);Pareto(indice_gra2,5);Pareto(indice_codas2,5);Pareto(indice_mabac2,5);Pareto(indice_vikor2,5);Pareto(indice_promethee2,5)],...
'VariableNames',{'Metodo','Rx','Ry','CL','Entropia','SSIM'},'RowNames',{'SMARTER','Fuzzy','TOPSIS','GRA','CODAS','MABAC','VIKOR','PROMETHEE II'});
writetable(T, strcat(path, "Data/Test ", testnum, "/decisiones_w2.xlsx"), 'Sheet',1,'Range','A1:H10000');


T = table({'SMARTER';'Fuzzy';'TOPSIS';'GRA';'CODAS';'MABAC';'VIKOR';'PROMETHEE II'},...
[Pareto(indice_smarter3,1);Pareto(indice_fuzzy3,1);Pareto(indice_topsis3,1);Pareto(indice_gra3,1);Pareto(indice_codas3,1);Pareto(indice_mabac3,1);Pareto(indice_vikor3,1);Pareto(indice_promethee3,1)],...
[Pareto(indice_smarter3,2);Pareto(indice_fuzzy3,2);Pareto(indice_topsis3,2);Pareto(indice_gra3,2);Pareto(indice_codas3,2);Pareto(indice_mabac3,2);Pareto(indice_vikor3,2);Pareto(indice_promethee3,2)],...
[Pareto(indice_smarter3,3);Pareto(indice_fuzzy3,3);Pareto(indice_topsis3,3);Pareto(indice_gra3,3);Pareto(indice_codas3,3);Pareto(indice_mabac3,3);Pareto(indice_vikor3,3);Pareto(indice_promethee3,3)],...
[Pareto(indice_smarter3,4);Pareto(indice_fuzzy3,4);Pareto(indice_topsis3,4);Pareto(indice_gra3,4);Pareto(indice_codas3,4);Pareto(indice_mabac3,4);Pareto(indice_vikor3,4);Pareto(indice_promethee3,4)],...
[Pareto(indice_smarter3,5);Pareto(indice_fuzzy3,5);Pareto(indice_topsis3,5);Pareto(indice_gra3,5);Pareto(indice_codas3,5);Pareto(indice_mabac3,5);Pareto(indice_vikor3,5);Pareto(indice_promethee3,5)],...
'VariableNames',{'Metodo','Rx','Ry','CL','Entropia','SSIM'},'RowNames',{'SMARTER','Fuzzy','TOPSIS','GRA','CODAS','MABAC','VIKOR','PROMETHEE II'});
writetable(T, strcat(path, "Data/Test ", testnum, "/decisiones_w3.xlsx"), 'Sheet',1,'Range','A1:H10000');

save(strcat(path, "Data/Test ", testnum, "/script.mat"));
