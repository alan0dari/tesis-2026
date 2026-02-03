function [] = script2(path)

load(strcat(path,'script.mat'));

[m2,n2]=size(Particulas);

fig_pareto = figure('WindowState','maximized','Name','Frente Pareto','NumberTitle','off');
axes1 = axes;
hold(axes1,'on');
scatter(Particulas(:,4), Particulas(:,5));
hold on;
scatter(Pareto(:,4),Pareto(:,5),25,'MarkerEdgeColor','r','MarkerFaceColor','r','LineWidth',1.5);
xlabel('Función Objetivo 1','FontWeight','bold','FontSize',24,'FontName','Lucida Bright');
ylabel('Función Objetivo 2','FontWeight','bold','FontSize',24,'FontName','Lucida Bright');
legend({'Partículas', 'Frente Pareto'});
set(legend,'Location','southwest','FontAngle','italic','FontSize',14,'FontName','Lucida Bright');
set(axes1,'FontName','Lucida Bright','FontSize',18);
filename = strcat(path,'pareto2.svg');
filename2 = strcat(path,'pareto2.fig');
saveas(fig_pareto,filename);
saveas(fig_pareto,filename2);

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
filename = strcat(path,'selecciones_metodos_w1.svg');
filename2 = strcat(path,'selecciones_metodos_w1.fig');
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
filename = strcat(path,'selecciones_metodos_w2.svg');
filename2 = strcat(path,'selecciones_metodos_w2.fig');
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
filename = strcat(path,'selecciones_metodos_w3.svg');
filename2 = strcat(path,'selecciones_metodos_w3.fig');
saveas(fig_decision3,filename);
saveas(fig_decision3,filename2);


T = table(Names, Pareto(:,1), Pareto(:,2), Pareto(:,3), Pareto(:,4), Pareto(:,5),'VariableNames',{'Nombre','Rx','Ry','CL','Entropia','SSIM'});
writetable(T, strcat(path,'pareto.xlsx'), 'Sheet',1,'Range','A1:H10000');


T = table({'SMARTER';'Fuzzy';'TOPSIS';'GRA';'CODAS';'MABAC';'VIKOR';'PROMETHEE II'},...
[Pareto(indice_smarter1,1);Pareto(indice_fuzzy1,1);Pareto(indice_topsis1,1);Pareto(indice_gra1,1);Pareto(indice_codas1,1);Pareto(indice_mabac1,1);Pareto(indice_vikor1,1);Pareto(indice_promethee1,1)],...
[Pareto(indice_smarter1,2);Pareto(indice_fuzzy1,2);Pareto(indice_topsis1,2);Pareto(indice_gra1,2);Pareto(indice_codas1,2);Pareto(indice_mabac1,2);Pareto(indice_vikor1,2);Pareto(indice_promethee1,2)],...
[Pareto(indice_smarter1,3);Pareto(indice_fuzzy1,3);Pareto(indice_topsis1,3);Pareto(indice_gra1,3);Pareto(indice_codas1,3);Pareto(indice_mabac1,3);Pareto(indice_vikor1,3);Pareto(indice_promethee1,3)],...
[Pareto(indice_smarter1,4);Pareto(indice_fuzzy1,4);Pareto(indice_topsis1,4);Pareto(indice_gra1,4);Pareto(indice_codas1,4);Pareto(indice_mabac1,4);Pareto(indice_vikor1,4);Pareto(indice_promethee1,4)],...
[Pareto(indice_smarter1,5);Pareto(indice_fuzzy1,5);Pareto(indice_topsis1,5);Pareto(indice_gra1,5);Pareto(indice_codas1,5);Pareto(indice_mabac1,5);Pareto(indice_vikor1,5);Pareto(indice_promethee1,5)],...
'VariableNames',{'Metodo','Rx','Ry','CL','Entropia','SSIM'},'RowNames',{'SMARTER','Fuzzy','TOPSIS','GRA','CODAS','MABAC','VIKOR','PROMETHEE II'});
writetable(T, strcat(path,'decisiones_w1.xlsx'), 'Sheet',1,'Range','A1:H10000');


T = table({'SMARTER';'Fuzzy';'TOPSIS';'GRA';'CODAS';'MABAC';'VIKOR';'PROMETHEE II'},...
[Pareto(indice_smarter2,1);Pareto(indice_fuzzy2,1);Pareto(indice_topsis2,1);Pareto(indice_gra2,1);Pareto(indice_codas2,1);Pareto(indice_mabac2,1);Pareto(indice_vikor2,1);Pareto(indice_promethee2,1)],...
[Pareto(indice_smarter2,2);Pareto(indice_fuzzy2,2);Pareto(indice_topsis2,2);Pareto(indice_gra2,2);Pareto(indice_codas2,2);Pareto(indice_mabac2,2);Pareto(indice_vikor2,2);Pareto(indice_promethee2,2)],...
[Pareto(indice_smarter2,3);Pareto(indice_fuzzy2,3);Pareto(indice_topsis2,3);Pareto(indice_gra2,3);Pareto(indice_codas2,3);Pareto(indice_mabac2,3);Pareto(indice_vikor2,3);Pareto(indice_promethee2,3)],...
[Pareto(indice_smarter2,4);Pareto(indice_fuzzy2,4);Pareto(indice_topsis2,4);Pareto(indice_gra2,4);Pareto(indice_codas2,4);Pareto(indice_mabac2,4);Pareto(indice_vikor2,4);Pareto(indice_promethee2,4)],...
[Pareto(indice_smarter2,5);Pareto(indice_fuzzy2,5);Pareto(indice_topsis2,5);Pareto(indice_gra2,5);Pareto(indice_codas2,5);Pareto(indice_mabac2,5);Pareto(indice_vikor2,5);Pareto(indice_promethee2,5)],...
'VariableNames',{'Metodo','Rx','Ry','CL','Entropia','SSIM'},'RowNames',{'SMARTER','Fuzzy','TOPSIS','GRA','CODAS','MABAC','VIKOR','PROMETHEE II'});
writetable(T, strcat(path,'decisiones_w2.xlsx'), 'Sheet',1,'Range','A1:H10000');


T = table({'SMARTER';'Fuzzy';'TOPSIS';'GRA';'CODAS';'MABAC';'VIKOR';'PROMETHEE II'},...
[Pareto(indice_smarter3,1);Pareto(indice_fuzzy3,1);Pareto(indice_topsis3,1);Pareto(indice_gra3,1);Pareto(indice_codas3,1);Pareto(indice_mabac3,1);Pareto(indice_vikor3,1);Pareto(indice_promethee3,1)],...
[Pareto(indice_smarter3,2);Pareto(indice_fuzzy3,2);Pareto(indice_topsis3,2);Pareto(indice_gra3,2);Pareto(indice_codas3,2);Pareto(indice_mabac3,2);Pareto(indice_vikor3,2);Pareto(indice_promethee3,2)],...
[Pareto(indice_smarter3,3);Pareto(indice_fuzzy3,3);Pareto(indice_topsis3,3);Pareto(indice_gra3,3);Pareto(indice_codas3,3);Pareto(indice_mabac3,3);Pareto(indice_vikor3,3);Pareto(indice_promethee3,3)],...
[Pareto(indice_smarter3,4);Pareto(indice_fuzzy3,4);Pareto(indice_topsis3,4);Pareto(indice_gra3,4);Pareto(indice_codas3,4);Pareto(indice_mabac3,4);Pareto(indice_vikor3,4);Pareto(indice_promethee3,4)],...
[Pareto(indice_smarter3,5);Pareto(indice_fuzzy3,5);Pareto(indice_topsis3,5);Pareto(indice_gra3,5);Pareto(indice_codas3,5);Pareto(indice_mabac3,5);Pareto(indice_vikor3,5);Pareto(indice_promethee3,5)],...
'VariableNames',{'Metodo','Rx','Ry','CL','Entropia','SSIM'},'RowNames',{'SMARTER','Fuzzy','TOPSIS','GRA','CODAS','MABAC','VIKOR','PROMETHEE II'});
writetable(T, strcat(path,'decisiones_w3.xlsx'), 'Sheet',1,'Range','A1:H10000');

save(strcat(path,'script.mat'));