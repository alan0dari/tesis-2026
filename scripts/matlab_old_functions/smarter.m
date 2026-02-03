function [eleccion, indice] = smarter(m,n,DM,w,MM)
%   %-- Método SMARTER
%
%   (update 24.06.2019)
%   Parámetros: 
%   D: Matriz de decisión de m alternativas con n criterios [mxn]
%   w: Matriz de pesos para los n criterio [1xn]
%   MM: Matriz de signos costo-beneficio de los n criterios[1xn]
%
%   Salida:
%   eleccion: Valor de ranking de la mejor alternativa
%   indice: Fila de la mejor alternativa
%   MN: Matriz de decisión D normalizada

F = Fun_DMnorm(m,n,DM,MM,3);    % Normalización de la matriz D con método Max-Min

F = F.*repmat(w,m,1); % Cálculo de rating

[eleccion, indice] = max(sum(F, 2));