load('DataFile1.mat')
i = 1; % frame
j = 1; % numero de sensor
k = 1; % numero de columna
feature = zeros(size(skeleton,1),139);
currentAction=1;
while(i <=size(skeleton,1))
        % Extraccion de los datos del esqueleto
        while(j <= 46)
            feature(i,k) = skeleton{i}(j,1);
            feature(i,k+1) = skeleton{i}(j,2);
            feature(i,k+2) = skeleton{i}(j,3);
            k = k+3;
            j = j+1;
        end
        j = 1;
        k = 1;
        if(i>Anotations(currentAction,2)) % Actualiza el class label para el conjunto actual
            currentAction=currentAction+1;
            classLabel=labels{currentAction,1};
        end
        % Escribir la label
        feature(i,139) = currentAction;
        i = i+1;
end
Datos1 = array2table(feature);
Datos1.Properties.VariableNames{'feature139'} = 'Clase';
Datos1(735:1118,:)=[];
Datos1(1694-384:2101-384,:)=[];
Datos1(2959-792:3513-792,:)=[];
Datos1(4223-1347:4766-1347,:)=[];
Datos1(5344-1891:6331-1891,:)=[];
Datos1(7057-2879:7542-2879,:)=[];
Datos1(8171-3365:8540-3365,:)=[];
Datos1(9094-3735:9502-3735,:)=[];
Datos1(9973-4144:10191-4144,:)=[];
Datos1(10367-4363:10678-4363,:)=[];
Datos1(11222-4675:11330-4675,:)=[];
Datos1(11885-4784:12527-4784,:)=[];



%csvwrite('Datos1.csv', Datos1);
writetable(Datos1,'DatabaseNoRepos.csv');
