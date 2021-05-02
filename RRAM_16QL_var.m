close all
clear all
clc

% Conductance weights
Goff = 1/900; %Siemens 
Gon = 1/110; %Siemens 

% Load all weight and bias matrices
b1 = load('b1.mat');
b1 = b1.b1;

b2 = load('b2.mat');
b2 = b2.b2;

b3 = load('b3.mat');
b3 = b3.b3;

o1 = load('o1.mat');
o1 = o1.o1;

o2 = load('o2.mat');
o2 = o2.o2;

o3 = load('o3.mat');
o3 = o3.o3;

pool1 = load('pool1.mat');
pool1 = pool1.pool1;

pool2 = load('pool2.mat');
pool2 = pool2.pool2;

pool3 = load('pool3.mat');
pool3 = pool3.pool3;

w1 = load('w1.mat');
w1 = w1.w1;

w2 = load('w2.mat');
w2 = w2.w2;

w3 = load('w3.mat');
w3 = w3.w3;

FC_bias_1 = load('FC_bias_1.mat');
FC_bias_1 = FC_bias_1.FC_bias_1;

FC_bias_2 = load('FC_bias_2.mat');
FC_bias_2 = FC_bias_2.FC_bias_2;

FC_output1 = load('FC_output1.mat');
FC_output1 = FC_output1.FC_output1;

FC_output2 = load('FC_output2.mat');
FC_output2 = FC_output2.FC_output2;

FC_weights_1 = load('FC_weights_1.mat');
FC_weights_1 = FC_weights_1.FC_weights_1;

FC_weights_2 = load('FC_weights_2.mat');
FC_weights_2 = FC_weights_2.FC_weights_2;

total_accuracy_cond = 0;


%% First Layer inference 
k = dir('C:\Caffe\Cifar10Data\cifar10Test');
N = length(k)-2 ; % Number of subfolders = number of classes 

labels = ["airplane"; "automobile"; "bird"; "cat"; "deer"; "dog"; "frog";... 
            "horse"; "ship"; "truck"];

for m = 1: N
    if m == 1
        Directory = 'C:\Caffe\Cifar10Data\cifar10Test\airplane'; 
    elseif m == 2
        Directory = 'C:\Caffe\Cifar10Data\cifar10Test\automobile';
    elseif m == 3
        Directory = 'C:\Caffe\Cifar10Data\cifar10Test\bird'; 
    elseif m == 4
        Directory = 'C:\Caffe\Cifar10Data\cifar10Test\cat';
    elseif m == 5
        Directory = 'C:\Caffe\Cifar10Data\cifar10Test\deer'; 
    elseif m == 6
        Directory = 'C:\Caffe\Cifar10Data\cifar10Test\dog';
    elseif m == 7
        Directory = 'C:\Caffe\Cifar10Data\cifar10Test\frog'; 
    elseif m == 8
        Directory = 'C:\Caffe\Cifar10Data\cifar10Test\horse';
    elseif m == 9
        Directory = 'C:\Caffe\Cifar10Data\cifar10Test\ship'; 
    else
        Directory = 'C:\Caffe\Cifar10Data\cifar10Test\truck';        
    end   
        
    Images2 = dir(Directory); 
    for n = 3: length(Images2)
        thisname = Images2(n).name;
        thisfile = fullfile(Directory, thisname);
        im_data = caffe.io.load_image(thisfile);
        im_data = imresize(im_data, [32 32]); 
        mean_data = caffe.io.read_mean('C:/caffe/examples/cifar10/mean.binaryproto');
        im_data = im_data - mean_data;
 
        d = im_data;
        
        filter_dim_1 = size(w1);
        numOfchannels_perFilter1 = filter_dim_1(1,3); % 3 for RGB
        numOfFilters1 = filter_dim_1(1,4);
        pad1 = 2;

        data_padded1 = padarray(d,[pad1 pad1],0,'both');

        % Second Layer Convolution
        filter_dim_2 = size(w2);
        numOfchannels_perFilter2 = filter_dim_2(1,3); % 3 for RGB
        numOfFilters2 = filter_dim_2(1,4);
        pad2 = 2;
       
        % Third Layer Convolution
        filter_dim_3 = size(w3);
        numOfchannels_perFilter3 = filter_dim_3(1,3); % 3 for RGB
        numOfFilters3 = filter_dim_3(1,4);
        pad3 = 2;

        %% Convert Convolution to Matrix Multiplication and Perform shifting for Negative Weights
        % For first layer structure, we will have 3 arrays each for a channel
        % (R,G,B). The 'R' array has the 32 filter only 'R' component and so on for
        % the rest. So the variable will be w1(:,:,1,j) for R, w1(:,:,2,j) for G and
        % w1(:,:,3,j) for B. And for j = 1, I will add the ouput and bias. Bias for
        % all filters will be included only in one array, so the dimension will be 
        % R(26x64) G(25x64) B(25x64). As for the 64, its bcz we have 32 filters and
        % we need 32 minimum for them  

        R_array = zeros(25,32);
        G_array = zeros(25,32);
        B_array = zeros(25,32);

        %Re-arrange the filters 
        for j = 1 : numOfFilters1
                R_array(:,j) = reshape(w1(:,:,1,j), 1,25);
                G_array(:,j) = reshape(w1(:,:,2,j), 1,25);
                B_array(:,j) = reshape(w1(:,:,3,j), 1,25);
        end
        % Concatenate only the first array with the bias
        R_array = [R_array; b1']; 

        im_r = cell(1,1);
        im_g = cell(1,1);
        im_b = cell(1,1);

        row1 = 1;
        col1 = 5;
        row2 = 1;
        col2 = 5;

        for i = 1: numOfchannels_perFilter1 %across all channels of a filter
            counter = 1;
            for s1 = 0:(size(data_padded1,1)-col1) %controls the shifting of the row and col per col
                for s2 = 0:(size(data_padded1,1)-col1) %controls the shifting of the col
                    input = data_padded1(row1+s1:col1+s1,row2+s2:col2+s2,i);
                    input = reshape(input,1,25);
                    if i == 1
                        im_r{counter,1} = input;
                    elseif i == 2
                        im_g{counter,1} = input;
                    else
                        im_b{counter,1} = input;
                    end
                    counter = counter + 1;
                end
            end
        end  

        %Each image is a row and each filter is a column
        im_r = cell2mat(im_r);
        im_r = [im_r,ones(length(im_r),1)];
        im_g = cell2mat(im_g);
        im_b = cell2mat(im_b);

        shift_r = zeros(numOfFilters1,1);
        shift_g = zeros(numOfFilters1,1);
        shift_b = zeros(numOfFilters1,1);

        for j = 1 : numOfFilters1 %across all filters
            shift_r(j) = min(R_array(:,j));
            shift_g(j) = min(G_array(:,j));
            shift_b(j) = min(B_array(:,j));
        end

        for j = 1 : numOfFilters1 %across all filters
            R_array(:,j) = R_array(:,j) -  shift_r(j);
            G_array(:,j) = G_array(:,j) -  shift_g(j);
            B_array(:,j) = B_array(:,j) -  shift_b(j);
        end

        % Create the shifting column and concatenate it
        R_array = [R_array, abs(shift_r'.*ones(size(R_array,1),size(R_array,2)))];
        G_array = [G_array, abs(shift_g'.*ones(size(G_array,1),size(G_array,2)))];
        B_array = [B_array, abs(shift_b'.*ones(size(B_array,1),size(B_array,2)))];

        minValue = min([R_array(:);G_array(:);B_array(:)]);
        maxValue = max([R_array(:);G_array(:);B_array(:)]);

        a_h = (Gon - Goff)/(maxValue - minValue);
        b_h = Gon - a_h *(maxValue);
        G_R_array = a_h .* R_array + b_h;
        G_G_array = a_h .* G_array + b_h;
        G_B_array = a_h .* B_array + b_h;
        
        out_r2 = im_r*R_array;
        out_g2 = im_g*G_array;
        out_b2 = im_b*B_array;

        output_first_layer_with_shifting = out_r2 + out_g2 + out_b2 ;
        shifting_matrix = output_first_layer_with_shifting(:,33:end);
        output_first_layer_with_shifting = output_first_layer_with_shifting(:,1:32);
        
        q_level = 16;
        delta = (Gon-Goff)/q_level;
        edges = cell(1,1);
        for i = 0:q_level
            edges{i+1} = Goff+(i*delta);
        end
        edges = cell2mat(edges);
        
        new_val_r = cell(1,1);
        new_val_g = cell(1,1);
        new_val_b = cell(1,1);
        
        variation = 0.01;        
        
        for i = 1: length(edges)-1
            range_mean = [edges(i) edges(i+1)];
            pd = makedist('normal','mu', mean(range_mean), 'sigma', mean(range_mean) + (variation*mean(range_mean)));
            upper = mean(range_mean) + (variation*mean(range_mean));
            lower = mean(range_mean) - (variation*mean(range_mean));
            t = truncate(pd, lower,upper);
            %without variation 
%             new_val_r{i} = mean(range_mean);
%             new_val_g{i} = mean(range_mean);
%             new_val_b{i} = mean(range_mean);
            %with variation 
            new_val_r{i} = random(t,1,1);
            new_val_g{i} = random(t,1,1);
            new_val_b{i} = random(t,1,1);

        end
        new_val_r = cell2mat(new_val_r);
        new_val_g = cell2mat(new_val_g);
        new_val_b = cell2mat(new_val_b);
                                       
         %QUANTIZED ARRAYS
        G_R_array_q = zeros(size(G_R_array,1),size(G_R_array,2));
        G_G_array_q = zeros(size(G_G_array,1),size(G_G_array,2));
        G_B_array_q = zeros(size(G_B_array,1),size(G_B_array,2));

         for i = 1:size(G_R_array,1)
                for j = 1:size(G_R_array,2)                    
                    if G_R_array(i,j) < edges(1) 
                        G_R_array_q(i,j) = new_val_r(1);

                    elseif G_R_array(i,j) >= edges(1) && G_R_array(i,j) < edges(2)
                        G_R_array_q(i,j) = new_val_r(1);

                    elseif G_R_array(i,j) >= edges(2) && G_R_array(i,j) < edges(3)
                        G_R_array_q(i,j) = new_val_r(2);     

                    elseif G_R_array(i,j) >= edges(3) && G_R_array(i,j) < edges(4)
                        G_R_array_q(i,j) = new_val_r(3);

                    elseif G_R_array(i,j) >= edges(4) && G_R_array(i,j) < edges(5)
                        G_R_array_q(i,j) = new_val_r(4); 

                    elseif G_R_array(i,j) >= edges(5) && G_R_array(i,j) < edges(6)
                        G_R_array_q(i,j) = new_val_r(5);             

                    elseif G_R_array(i,j) >= edges(6) && G_R_array(i,j) < edges(7)
                        G_R_array_q(i,j) = new_val_r(6);   

                    elseif G_R_array(i,j) >= edges(7) && G_R_array(i,j) < edges(8)
                        G_R_array_q(i,j) = new_val_r(7);          

                    elseif G_R_array(i,j) >= edges(8) && G_R_array(i,j) < edges(9)
                        G_R_array_q(i,j) = new_val_r(8);              

                    elseif G_R_array(i,j) >= edges(9) && G_R_array(i,j) < edges(10) 
                         G_R_array_q(i,j) = new_val_r(9); 

                    elseif G_R_array(i,j) >= edges(10) && G_R_array(i,j) < edges(11)
                         G_R_array_q(i,j) = new_val_r(10); 

                    elseif G_R_array(i,j) >= edges(11) && G_R_array(i,j) < edges(12)
                         G_R_array_q(i,j) = new_val_r(11); 

                    elseif G_R_array(i,j) >= edges(12) && G_R_array(i,j) < edges(13)
                         G_R_array_q(i,j) = new_val_r(12); 

                    elseif G_R_array(i,j) >= edges(13) && G_R_array(i,j) < edges(14) 
                         G_R_array_q(i,j) = new_val_r(13); 

                    elseif G_R_array(i,j) >= edges(14) && G_R_array(i,j) < edges(15) 
                         G_R_array_q(i,j) = new_val_r(14); 

                    elseif G_R_array(i,j) >= edges(15) && G_R_array(i,j) < edges(16) 
                         G_R_array_q(i,j) = new_val_r(15); 

                    elseif G_R_array(i,j) >= edges(16) && G_R_array(i,j) <= edges(17)
                         G_R_array_q(i,j) = new_val_r(16);                 
                    end
                end

        end

         for i = 1:size(G_G_array,1)
                for j = 1:size(G_G_array,2)                    
                    if G_G_array(i,j) < edges(1) 
                        G_G_array_q(i,j) = new_val_g(1);

                    elseif G_G_array(i,j) >= edges(1) && G_G_array(i,j) < edges(2)
                        G_G_array_q(i,j) = new_val_g(1);

                    elseif G_G_array(i,j) >= edges(2) && G_G_array(i,j) < edges(3)
                        G_G_array_q(i,j) = new_val_g(2);     

                    elseif G_G_array(i,j) >= edges(3) && G_G_array(i,j) < edges(4)
                        G_G_array_q(i,j) = new_val_g(3);

                    elseif G_G_array(i,j) >= edges(4) && G_G_array(i,j) < edges(5)
                        G_G_array_q(i,j) = new_val_g(4); 

                    elseif G_G_array(i,j) >= edges(5) && G_G_array(i,j) < edges(6)
                        G_G_array_q(i,j) = new_val_g(5);             

                    elseif G_G_array(i,j) >= edges(6) && G_G_array(i,j) < edges(7)
                        G_G_array_q(i,j) = new_val_g(6);   

                    elseif G_G_array(i,j) >= edges(7) && G_G_array(i,j) < edges(8)
                        G_G_array_q(i,j) = new_val_g(7);          

                    elseif G_G_array(i,j) >= edges(8) && G_G_array(i,j) < edges(9)
                        G_G_array_q(i,j) = new_val_g(8);              

                    elseif G_G_array(i,j) >= edges(9) && G_G_array(i,j) < edges(10) 
                         G_G_array_q(i,j) = new_val_g(9); 

                    elseif G_G_array(i,j) >= edges(10) && G_G_array(i,j) < edges(11)
                         G_G_array_q(i,j) = new_val_g(10); 

                    elseif G_G_array(i,j) >= edges(11) && G_G_array(i,j) < edges(12)
                         G_G_array_q(i,j) = new_val_g(11); 

                    elseif G_G_array(i,j) >= edges(12) && G_G_array(i,j) < edges(13)
                         G_G_array_q(i,j) = new_val_g(12); 

                    elseif G_G_array(i,j) >= edges(13) && G_G_array(i,j) < edges(14) 
                         G_G_array_q(i,j) = new_val_g(13); 

                    elseif G_G_array(i,j) >= edges(14) && G_G_array(i,j) < edges(15) 
                         G_G_array_q(i,j) = new_val_g(14); 

                    elseif G_G_array(i,j) >= edges(15) && G_G_array(i,j) < edges(16) 
                         G_G_array_q(i,j) = new_val_g(15); 

                    elseif G_G_array(i,j) >= edges(16) && G_G_array(i,j) <= edges(17)
                         G_G_array_q(i,j) = new_val_g(16);                 
                    end
                end

         end

         for i = 1:size(G_B_array,1)
                for j = 1:size(G_B_array,2)                    
                    if G_B_array(i,j) < edges(1) 
                        G_B_array_q(i,j) = new_val_b(1);

                    elseif G_B_array(i,j) >= edges(1) && G_B_array(i,j) < edges(2)
                        G_B_array_q(i,j) = new_val_b(1);

                    elseif G_B_array(i,j) >= edges(2) && G_B_array(i,j) < edges(3)
                        G_B_array_q(i,j) = new_val_b(2);     

                    elseif G_B_array(i,j) >= edges(3) && G_B_array(i,j) < edges(4)
                        G_B_array_q(i,j) = new_val_b(3);

                    elseif G_B_array(i,j) >= edges(4) && G_B_array(i,j) < edges(5)
                        G_B_array_q(i,j) = new_val_b(4); 

                    elseif G_B_array(i,j) >= edges(5) && G_B_array(i,j) < edges(6)
                        G_B_array_q(i,j) = new_val_b(5);             

                    elseif G_B_array(i,j) >= edges(6) && G_B_array(i,j) < edges(7)
                        G_B_array_q(i,j) = new_val_b(6);   

                    elseif G_B_array(i,j) >= edges(7) && G_B_array(i,j) < edges(8)
                        G_B_array_q(i,j) = new_val_b(7);          

                    elseif G_B_array(i,j) >= edges(8) && G_B_array(i,j) < edges(9)
                        G_B_array_q(i,j) = new_val_b(8);              

                    elseif G_B_array(i,j) >= edges(9) && G_B_array(i,j) < edges(10) 
                         G_B_array_q(i,j) = new_val_b(9); 

                    elseif G_B_array(i,j) >= edges(10) && G_B_array(i,j) < edges(11)
                         G_B_array_q(i,j) = new_val_b(10); 

                    elseif G_B_array(i,j) >= edges(11) && G_B_array(i,j) < edges(12)
                         G_B_array_q(i,j) = new_val_b(11); 

                    elseif G_B_array(i,j) >= edges(12) && G_B_array(i,j) < edges(13)
                         G_B_array_q(i,j) = new_val_b(12); 

                    elseif G_B_array(i,j) >= edges(13) && G_B_array(i,j) < edges(14) 
                         G_B_array_q(i,j) = new_val_b(13); 

                    elseif G_B_array(i,j) >= edges(14) && G_B_array(i,j) < edges(15) 
                         G_B_array_q(i,j) = new_val_b(14); 

                    elseif G_B_array(i,j) >= edges(15) && G_B_array(i,j) < edges(16) 
                         G_B_array_q(i,j) = new_val_b(15); 

                    elseif G_B_array(i,j) >= edges(16) && G_B_array(i,j) <= edges(17)
                         G_B_array_q(i,j) = new_val_b(16);                 
                    end
                end

         end               
        
        out_r2_cond = im_r*G_R_array_q;
        out_g2_cond = im_g*G_G_array_q;
        out_b2_cond = im_b*G_B_array_q;

        output_first_layer_with_shifting_cond = out_r2_cond + out_g2_cond + out_b2_cond ;
        shifting_matrix_cond = output_first_layer_with_shifting_cond(:,33:end);
        output_first_layer_with_shifting_cond = output_first_layer_with_shifting_cond(:,1:32);

        % Each col is an output filter (its flattening happened row-by-row)
        output_after_shifting = output_first_layer_with_shifting - shifting_matrix;

        output_first_layer_with_shifting_cond = output_first_layer_with_shifting_cond - shifting_matrix_cond;

        % Re-shape it to pass it through pooling and relu
        output_1_shifting = permute(reshape(output_after_shifting,[32,32,32]),[2 1 3]);
        output_1_cond = permute(reshape(output_first_layer_with_shifting_cond,[32,32,32]),[2 1 3]);

        PoolMap1_shifting = zeros(size(pool1,1),size(pool1,2), size(pool1,3));
        PoolMap1_shifting_cond = zeros(size(pool1,1),size(pool1,2), size(pool1,3));

        row1 = 1;
        col1 = 3;
        row2 = 1;
        col2 = 3;
        for j = 1 : numOfFilters1 %across all filters
            k1 = 1;
            for s1 = 0:2:(size(output_1_shifting,1)-2) %controls the shifting of the row and col per col
                k2 = 1;
                for s2 = 0:2:(size(output_1_shifting,1)-2) %controls the shifting of the col
                    if (s1 == (size(output_1_shifting,1)-2)) && (s2 == (size(output_1_shifting,1)-2))
                        out1 = max(max(max(output_1_shifting(row1+s1:col1+s1-1,row2+s2:col2+s2-1,j),0)));
                    elseif s2 == (size(output_1_shifting,1)-2)
                        out1 = max(max(max(output_1_shifting(row1+s1:col1+s1,row2+s2:col2+s2-1,j),0))); 
                    elseif s1 == (size(output_1_shifting,1)-2)
                        out1 = max(max(max(output_1_shifting(row1+s1:col1+s1-1,row2+s2:col2+s2,j),0))); 
                    else 
                        out1 = max(max(max(output_1_shifting(row1+s1:col1+s1,row2+s2:col2+s2,j),0)));  
                    end    
                    PoolMap1_shifting(k1,k2,j) = out1;
                    k2 = k2 + 1;

                end
                k1 = k1 + 1;
            end  
        end

        for j = 1 : numOfFilters1 %across all filters
            k1 = 1;
            for s1 = 0:2:(size(output_1_cond,1)-2) %controls the shifting of the row and col per col
                k2 = 1;
                for s2 = 0:2:(size(output_1_cond,1)-2) %controls the shifting of the col
                    if (s1 == (size(output_1_cond,1)-2)) && (s2 == (size(output_1_cond,1)-2))
                        out1 = max(max(max(output_1_cond(row1+s1:col1+s1-1,row2+s2:col2+s2-1,j),0)));
                    elseif s2 == (size(output_1_cond,1)-2)
                        out1 = max(max(max(output_1_cond(row1+s1:col1+s1,row2+s2:col2+s2-1,j),0))); 
                    elseif s1 == (size(output_1_cond,1)-2)
                        out1 = max(max(max(output_1_cond(row1+s1:col1+s1-1,row2+s2:col2+s2,j),0))); 
                    else 
                        out1 = max(max(max(output_1_cond(row1+s1:col1+s1,row2+s2:col2+s2,j),0)));  
                    end    
                    PoolMap1_shifting_cond(k1,k2,j) = out1;
                    k2 = k2 + 1;

                end
                k1 = k1 + 1;
            end  
        end

        offset_1 = 46.9995; 
        PoolMap1_shifting_cond = PoolMap1_shifting_cond * offset_1;

        % % Reshape the data again to pass it through the second MAC (ConV)
        % % Flatten the filters where each f1 from every channel becomes a column in
        % % f2_1

        data_padded2_shift = padarray(PoolMap1_shifting,[pad2 pad2],0,'both');
                data_padded2_shift_cond = padarray(PoolMap1_shifting_cond,[pad2 pad2],0,'both');

        % data_padded2_shift = padarray(PoolMap1,[pad2 pad2],0,'both');
        f2_1 = zeros(25,32); 
        f2_2 = zeros(25,32); 
        f2_3 = zeros(25,32); 
        f2_4 = zeros(25,32); 
        f2_5 = zeros(25,32); 
        f2_6 = zeros(25,32); 
        f2_7 = zeros(25,32); 
        f2_8 = zeros(25,32); 
        f2_9 = zeros(25,32); 
        f2_10 = zeros(25,32); 
        f2_11 = zeros(25,32); 
        f2_12 = zeros(25,32); 
        f2_13 = zeros(25,32); 
        f2_14 = zeros(25,32); 
        f2_15 = zeros(25,32); 
        f2_16 = zeros(25,32); 
        f2_17 = zeros(25,32); 
        f2_18 = zeros(25,32); 
        f2_19 = zeros(25,32); 
        f2_20 = zeros(25,32); 
        f2_21 = zeros(25,32); 
        f2_22 = zeros(25,32); 
        f2_23 = zeros(25,32); 
        f2_24 = zeros(25,32); 
        f2_25 = zeros(25,32); 
        f2_26 = zeros(25,32); 
        f2_27 = zeros(25,32); 
        f2_28 = zeros(25,32); 
        f2_29 = zeros(25,32); 
        f2_30 = zeros(25,32); 
        f2_31 = zeros(25,32); 
        f2_32 = zeros(25,32); 

        for j = 1 : numOfFilters2
            f2_1(:,j) = reshape(w2(:,:,1,j), 1,25);
            f2_2(:,j) = reshape(w2(:,:,2,j), 1,25);
            f2_3(:,j) = reshape(w2(:,:,3,j), 1,25);
            f2_4(:,j) = reshape(w2(:,:,4,j), 1,25);
            f2_5(:,j) = reshape(w2(:,:,5,j), 1,25);
            f2_6(:,j) = reshape(w2(:,:,6,j), 1,25);
            f2_7(:,j) = reshape(w2(:,:,7,j), 1,25);
            f2_8(:,j) = reshape(w2(:,:,8,j), 1,25);
            f2_9(:,j) = reshape(w2(:,:,9,j), 1,25);
            f2_10(:,j) = reshape(w2(:,:,10,j), 1,25);
            f2_11(:,j) = reshape(w2(:,:,11,j), 1,25);
            f2_12(:,j) = reshape(w2(:,:,12,j), 1,25);
            f2_13(:,j) = reshape(w2(:,:,13,j), 1,25);
            f2_14(:,j) = reshape(w2(:,:,14,j), 1,25);
            f2_15(:,j) = reshape(w2(:,:,15,j), 1,25);
            f2_16(:,j) = reshape(w2(:,:,16,j), 1,25);
            f2_17(:,j) = reshape(w2(:,:,17,j), 1,25);
            f2_18(:,j) = reshape(w2(:,:,18,j), 1,25);
            f2_19(:,j) = reshape(w2(:,:,19,j), 1,25);
            f2_20(:,j) = reshape(w2(:,:,20,j), 1,25);
            f2_21(:,j) = reshape(w2(:,:,21,j), 1,25);
            f2_22(:,j) = reshape(w2(:,:,22,j), 1,25);
            f2_23(:,j) = reshape(w2(:,:,23,j), 1,25);
            f2_24(:,j) = reshape(w2(:,:,24,j), 1,25);
            f2_25(:,j) = reshape(w2(:,:,25,j), 1,25);
            f2_26(:,j) = reshape(w2(:,:,26,j), 1,25);
            f2_27(:,j) = reshape(w2(:,:,27,j), 1,25);
            f2_28(:,j) = reshape(w2(:,:,28,j), 1,25);
            f2_29(:,j) = reshape(w2(:,:,29,j), 1,25);
            f2_30(:,j) = reshape(w2(:,:,30,j), 1,25);
            f2_31(:,j) = reshape(w2(:,:,31,j), 1,25);
            f2_32(:,j) = reshape(w2(:,:,32,j), 1,25);
        end
        % Concatenate only the first array with the bias
        f2_1 = [f2_1; b2'] ;

        im_1 = cell(1,1); % Each row wll be an image patch
        im_2 = cell(1,1);
        im_3 = cell(1,1);
        im_4 = cell(1,1);
        im_5 = cell(1,1);
        im_6 = cell(1,1);
        im_7 = cell(1,1);
        im_8 = cell(1,1);
        im_9 = cell(1,1);
        im_10 = cell(1,1);
        im_11 = cell(1,1);
        im_12 = cell(1,1);
        im_13 = cell(1,1);
        im_14 = cell(1,1);
        im_15 = cell(1,1);
        im_16 = cell(1,1);
        im_17 = cell(1,1);
        im_18 = cell(1,1);
        im_19 = cell(1,1);
        im_20 = cell(1,1);
        im_21 = cell(1,1);
        im_22 = cell(1,1);
        im_23 = cell(1,1);
        im_24 = cell(1,1);
        im_25 = cell(1,1);
        im_26 = cell(1,1);
        im_27 = cell(1,1);
        im_28 = cell(1,1);
        im_29 = cell(1,1);
        im_30 = cell(1,1);
        im_31 = cell(1,1);
        im_32 = cell(1,1);

        % Flatten image patches and put them in their associated arrays depending
        % on the channel

        row1 = 1;
        col1 = 5;
        row2 = 1;
        col2 = 5;

        for i = 1: numOfchannels_perFilter2 %across all channels of a filter
            counter = 1;
            for s1 = 0:(size(data_padded2_shift,1)-col1) %controls the shifting of the row and col per col
                for s2 = 0:(size(data_padded2_shift,1)-col1) %controls the shifting of the col
                    input = data_padded2_shift(row1+s1:col1+s1,row2+s2:col2+s2,i);
                    input = reshape(input,1,25);
                    if i == 1
                        im_1{counter,1} = input;
                    elseif i == 2
                        im_2{counter,1} = input;
                    elseif i == 3
                        im_3{counter,1} = input;
                    elseif i == 4
                        im_4{counter,1} = input;
                    elseif i == 5
                        im_5{counter,1} = input;                
                    elseif i == 6
                        im_6{counter,1} = input;
                    elseif i == 7
                        im_7{counter,1} = input;                
                    elseif i == 8
                        im_8{counter,1} = input;
                    elseif i == 9
                        im_9{counter,1} = input;                
                    elseif i == 10
                        im_10{counter,1} = input;
                    elseif i == 11
                        im_11{counter,1} = input;                
                    elseif i == 12
                        im_12{counter,1} = input;
                    elseif i == 13
                        im_13{counter,1} = input;                
                    elseif i == 14
                        im_14{counter,1} = input;
                    elseif i == 15
                        im_15{counter,1} = input;                
                    elseif i == 16
                        im_16{counter,1} = input;
                    elseif i == 17
                        im_17{counter,1} = input;                
                    elseif i == 18
                        im_18{counter,1} = input;
                    elseif i == 19
                        im_19{counter,1} = input;                
                    elseif i == 20
                        im_20{counter,1} = input;
                    elseif i == 21
                        im_21{counter,1} = input;                
                    elseif i == 22
                        im_22{counter,1} = input;
                    elseif i == 23
                        im_23{counter,1} = input;
                     elseif i == 24
                        im_24{counter,1} = input;
                    elseif i == 25
                        im_25{counter,1} = input;                
                    elseif i == 26
                        im_26{counter,1} = input;
                    elseif i == 27
                        im_27{counter,1} = input;                
                    elseif i == 28
                        im_28{counter,1} = input;
                    elseif i == 29
                        im_29{counter,1} = input;                
                    elseif i == 30
                        im_30{counter,1} = input;
                    elseif i == 31
                        im_31{counter,1} = input;                                                        
                    else
                        im_32{counter,1} = input;
                    end
                    counter = counter + 1;
                end
            end
        end  

        %Each image is a row and each filter is a column
        im_1 = cell2mat(im_1);
        im_1 = [im_1,ones(length(im_1),1)];

        im_2 = cell2mat(im_2);
        im_3 = cell2mat(im_3);
        im_4 = cell2mat(im_4);
        im_5 = cell2mat(im_5);
        im_6 = cell2mat(im_6);
        im_7 = cell2mat(im_7);
        im_8 = cell2mat(im_8);
        im_9 = cell2mat(im_9);
        im_10 = cell2mat(im_10);
        im_11 = cell2mat(im_11);
        im_12 = cell2mat(im_12);
        im_13 = cell2mat(im_13);
        im_14 = cell2mat(im_14);
        im_15 = cell2mat(im_15);
        im_16 = cell2mat(im_16);
        im_17 = cell2mat(im_17);
        im_18 = cell2mat(im_18);
        im_19 = cell2mat(im_19);
        im_20 = cell2mat(im_20);
        im_21 = cell2mat(im_21);
        im_22 = cell2mat(im_22);
        im_23 = cell2mat(im_23);
        im_24 = cell2mat(im_24);
        im_25 = cell2mat(im_25);
        im_26 = cell2mat(im_26);
        im_27 = cell2mat(im_27);
        im_28 = cell2mat(im_28);
        im_29 = cell2mat(im_29);
        im_30 = cell2mat(im_30);
        im_31 = cell2mat(im_31);
        im_32 = cell2mat(im_32);


        im_cond_1 = cell(1,1); % Each row wll be an image patch
        im_cond_2 = cell(1,1);
        im_cond_3 = cell(1,1);
        im_cond_4 = cell(1,1);
        im_cond_5 = cell(1,1);
        im_cond_6 = cell(1,1);
        im_cond_7 = cell(1,1);
        im_cond_8 = cell(1,1);
        im_cond_9 = cell(1,1);
        im_cond_10 = cell(1,1);
        im_cond_11 = cell(1,1);
        im_cond_12 = cell(1,1);
        im_cond_13 = cell(1,1);
        im_cond_14 = cell(1,1);
        im_cond_15 = cell(1,1);
        im_cond_16 = cell(1,1);
        im_cond_17 = cell(1,1);
        im_cond_18 = cell(1,1);
        im_cond_19 = cell(1,1);
        im_cond_20 = cell(1,1);
        im_cond_21 = cell(1,1);
        im_cond_22 = cell(1,1);
        im_cond_23 = cell(1,1);
        im_cond_24 = cell(1,1);
        im_cond_25 = cell(1,1);
        im_cond_26 = cell(1,1);
        im_cond_27 = cell(1,1);
        im_cond_28 = cell(1,1);
        im_cond_29 = cell(1,1);
        im_cond_30 = cell(1,1);
        im_cond_31 = cell(1,1);
        im_cond_32 = cell(1,1);

        for i = 1: numOfchannels_perFilter2 %across all channels of a filter
            counter = 1;
            for s1 = 0:(size(data_padded2_shift_cond,1)-col1) %controls the shifting of the row and col per col
                for s2 = 0:(size(data_padded2_shift_cond,1)-col1) %controls the shifting of the col
                    input = data_padded2_shift_cond(row1+s1:col1+s1,row2+s2:col2+s2,i);
                    input = reshape(input,1,25);
                    if i == 1
                        im_cond_1{counter,1} = input;
                    elseif i == 2
                        im_cond_2{counter,1} = input;
                    elseif i == 3
                        im_cond_3{counter,1} = input;
                    elseif i == 4
                        im_cond_4{counter,1} = input;
                    elseif i == 5
                        im_cond_5{counter,1} = input;                
                    elseif i == 6
                        im_cond_6{counter,1} = input;
                    elseif i == 7
                        im_cond_7{counter,1} = input;                
                    elseif i == 8
                        im_cond_8{counter,1} = input;
                    elseif i == 9
                        im_cond_9{counter,1} = input;                
                    elseif i == 10
                        im_cond_10{counter,1} = input;
                    elseif i == 11
                        im_cond_11{counter,1} = input;                
                    elseif i == 12
                        im_cond_12{counter,1} = input;
                    elseif i == 13
                        im_cond_13{counter,1} = input;                
                    elseif i == 14
                        im_cond_14{counter,1} = input;
                    elseif i == 15
                        im_cond_15{counter,1} = input;                
                    elseif i == 16
                        im_cond_16{counter,1} = input;
                    elseif i == 17
                        im_cond_17{counter,1} = input;                
                    elseif i == 18
                        im_cond_18{counter,1} = input;
                    elseif i == 19
                        im_cond_19{counter,1} = input;                
                    elseif i == 20
                        im_cond_20{counter,1} = input;
                    elseif i == 21
                        im_cond_21{counter,1} = input;                
                    elseif i == 22
                        im_cond_22{counter,1} = input;
                    elseif i == 23
                        im_cond_23{counter,1} = input;
                     elseif i == 24
                        im_cond_24{counter,1} = input;
                    elseif i == 25
                        im_cond_25{counter,1} = input;                
                    elseif i == 26
                        im_cond_26{counter,1} = input;
                    elseif i == 27
                        im_cond_27{counter,1} = input;                
                    elseif i == 28
                        im_cond_28{counter,1} = input;
                    elseif i == 29
                        im_cond_29{counter,1} = input;                
                    elseif i == 30
                        im_cond_30{counter,1} = input;
                    elseif i == 31
                        im_cond_31{counter,1} = input;                                                        
                    else
                        im_cond_32{counter,1} = input;
                    end
                    counter = counter + 1;
                end
            end
        end  

        %Each image is a row and each filter is a column
        im_cond_1 = cell2mat(im_cond_1);
        im_cond_1 = [im_cond_1,ones(length(im_cond_1),1)];

        im_cond_2 = cell2mat(im_cond_2);
        im_cond_3 = cell2mat(im_cond_3);
        im_cond_4 = cell2mat(im_cond_4);
        im_cond_5 = cell2mat(im_cond_5);
        im_cond_6 = cell2mat(im_cond_6);
        im_cond_7 = cell2mat(im_cond_7);
        im_cond_8 = cell2mat(im_cond_8);
        im_cond_9 = cell2mat(im_cond_9);
        im_cond_10 = cell2mat(im_cond_10);
        im_cond_11 = cell2mat(im_cond_11);
        im_cond_12 = cell2mat(im_cond_12);
        im_cond_13 = cell2mat(im_cond_13);
        im_cond_14 = cell2mat(im_cond_14);
        im_cond_15 = cell2mat(im_cond_15);
        im_cond_16 = cell2mat(im_cond_16);
        im_cond_17 = cell2mat(im_cond_17);
        im_cond_18 = cell2mat(im_cond_18);
        im_cond_19 = cell2mat(im_cond_19);
        im_cond_20 = cell2mat(im_cond_20);
        im_cond_21 = cell2mat(im_cond_21);
        im_cond_22 = cell2mat(im_cond_22);
        im_cond_23 = cell2mat(im_cond_23);
        im_cond_24 = cell2mat(im_cond_24);
        im_cond_25 = cell2mat(im_cond_25);
        im_cond_26 = cell2mat(im_cond_26);
        im_cond_27 = cell2mat(im_cond_27);
        im_cond_28 = cell2mat(im_cond_28);
        im_cond_29 = cell2mat(im_cond_29);
        im_cond_30 = cell2mat(im_cond_30);
        im_cond_31 = cell2mat(im_cond_31);
        im_cond_32 = cell2mat(im_cond_32);

        shift_1 = zeros(numOfFilters2,1);
        shift_2 = zeros(numOfFilters2,1);
        shift_3 = zeros(numOfFilters2,1);
        shift_4 = zeros(numOfFilters2,1);
        shift_5 = zeros(numOfFilters2,1);
        shift_6 = zeros(numOfFilters2,1);
        shift_7 = zeros(numOfFilters2,1);
        shift_8 = zeros(numOfFilters2,1);
        shift_9 = zeros(numOfFilters2,1);
        shift_10 = zeros(numOfFilters2,1);
        shift_11 = zeros(numOfFilters2,1);
        shift_12 = zeros(numOfFilters2,1);
        shift_13 = zeros(numOfFilters2,1);
        shift_14 = zeros(numOfFilters2,1);
        shift_15 = zeros(numOfFilters2,1);
        shift_16 = zeros(numOfFilters2,1);
        shift_17 = zeros(numOfFilters2,1);
        shift_18 = zeros(numOfFilters2,1);
        shift_19 = zeros(numOfFilters2,1);
        shift_20 = zeros(numOfFilters2,1);
        shift_21 = zeros(numOfFilters2,1);
        shift_22 = zeros(numOfFilters2,1);
        shift_23 = zeros(numOfFilters2,1);
        shift_24 = zeros(numOfFilters2,1);
        shift_25 = zeros(numOfFilters2,1);
        shift_26 = zeros(numOfFilters2,1);
        shift_27 = zeros(numOfFilters2,1);
        shift_28 = zeros(numOfFilters2,1);
        shift_29 = zeros(numOfFilters2,1);
        shift_30 = zeros(numOfFilters2,1);
        shift_31 = zeros(numOfFilters2,1);
        shift_32 = zeros(numOfFilters2,1);

        for j = 1 : numOfFilters2 %across all filters
            shift_1(j) = min(f2_1(:,j));
            shift_2(j) = min(f2_2(:,j));
            shift_3(j) = min(f2_3(:,j));
            shift_4(j) = min(f2_4(:,j));
            shift_5(j) = min(f2_5(:,j));
            shift_6(j) = min(f2_6(:,j));
            shift_7(j) = min(f2_7(:,j));
            shift_8(j) = min(f2_8(:,j));
            shift_9(j) = min(f2_9(:,j));
            shift_10(j) = min(f2_10(:,j));
            shift_11(j) = min(f2_11(:,j));
            shift_12(j) = min(f2_12(:,j));
            shift_13(j) = min(f2_13(:,j));
            shift_14(j) = min(f2_14(:,j));
            shift_15(j) = min(f2_15(:,j));
            shift_16(j) = min(f2_16(:,j));
            shift_17(j) = min(f2_17(:,j));
            shift_18(j) = min(f2_18(:,j));
            shift_19(j) = min(f2_19(:,j));
            shift_20(j) = min(f2_20(:,j));
            shift_21(j) = min(f2_21(:,j));
            shift_22(j) = min(f2_22(:,j));
            shift_23(j) = min(f2_23(:,j));
            shift_24(j) = min(f2_24(:,j));
            shift_25(j) = min(f2_25(:,j));
            shift_26(j) = min(f2_26(:,j));
            shift_27(j) = min(f2_27(:,j));
            shift_28(j) = min(f2_28(:,j));
            shift_29(j) = min(f2_29(:,j));
            shift_30(j) = min(f2_30(:,j));
            shift_31(j) = min(f2_31(:,j));
            shift_32(j) = min(f2_32(:,j));
        end

        for j = 1 : numOfFilters2 %across all filters
            f2_1(:,j) = f2_1(:,j) -  shift_1(j);
            f2_2(:,j) = f2_2(:,j) -  shift_2(j);
            f2_3(:,j) = f2_3(:,j) -  shift_3(j);
            f2_4(:,j) = f2_4(:,j) -  shift_4(j);
            f2_5(:,j) = f2_5(:,j) -  shift_5(j);
            f2_6(:,j) = f2_6(:,j) -  shift_6(j);
            f2_7(:,j) = f2_7(:,j) -  shift_7(j);
            f2_8(:,j) = f2_8(:,j) -  shift_8(j);
            f2_9(:,j) = f2_9(:,j) -  shift_9(j);
            f2_10(:,j) = f2_10(:,j) -  shift_10(j);
            f2_11(:,j) = f2_11(:,j) -  shift_11(j);
            f2_12(:,j) = f2_12(:,j) -  shift_12(j);
            f2_13(:,j) = f2_13(:,j) -  shift_13(j);
            f2_14(:,j) = f2_14(:,j) -  shift_14(j);
            f2_15(:,j) = f2_15(:,j) -  shift_15(j);
            f2_16(:,j) = f2_16(:,j) -  shift_16(j);
            f2_17(:,j) = f2_17(:,j) -  shift_17(j);
            f2_18(:,j) = f2_18(:,j) -  shift_18(j);
            f2_19(:,j) = f2_19(:,j) -  shift_19(j);
            f2_20(:,j) = f2_20(:,j) -  shift_20(j);
            f2_21(:,j) = f2_21(:,j) -  shift_21(j);
            f2_22(:,j) = f2_22(:,j) -  shift_22(j);
            f2_23(:,j) = f2_23(:,j) -  shift_23(j);
            f2_24(:,j) = f2_24(:,j) -  shift_24(j);
            f2_25(:,j) = f2_25(:,j) -  shift_25(j);
            f2_26(:,j) = f2_26(:,j) -  shift_26(j);
            f2_27(:,j) = f2_27(:,j) -  shift_27(j);
            f2_28(:,j) = f2_28(:,j) -  shift_28(j);
            f2_29(:,j) = f2_29(:,j) -  shift_29(j);
            f2_30(:,j) = f2_30(:,j) -  shift_30(j);
            f2_31(:,j) = f2_31(:,j) -  shift_31(j);
            f2_32(:,j) = f2_32(:,j) -  shift_32(j);
        end

        % Create the shifting column and concatenate it
        f2_1 = [f2_1, abs(shift_1'.*ones(size(f2_1,1),size(f2_1,2)))];
        f2_2 = [f2_2, abs(shift_2'.*ones(size(f2_2,1),size(f2_2,2)))];
        f2_3 = [f2_3, abs(shift_3'.*ones(size(f2_3,1),size(f2_3,2)))];
        f2_4 = [f2_4, abs(shift_4'.*ones(size(f2_4,1),size(f2_4,2)))];
        f2_5 = [f2_5, abs(shift_5'.*ones(size(f2_5,1),size(f2_5,2)))];
        f2_6 = [f2_6, abs(shift_6'.*ones(size(f2_6,1),size(f2_6,2)))];
        f2_7 = [f2_7, abs(shift_7'.*ones(size(f2_7,1),size(f2_7,2)))];
        f2_8 = [f2_8, abs(shift_8'.*ones(size(f2_8,1),size(f2_8,2)))];
        f2_9 = [f2_9, abs(shift_9'.*ones(size(f2_9,1),size(f2_9,2)))];
        f2_10 = [f2_10, abs(shift_10'.*ones(size(f2_10,1),size(f2_10,2)))];
        f2_11 = [f2_11, abs(shift_11'.*ones(size(f2_11,1),size(f2_11,2)))];
        f2_12 = [f2_12, abs(shift_12'.*ones(size(f2_12,1),size(f2_12,2)))];
        f2_13 = [f2_13, abs(shift_13'.*ones(size(f2_13,1),size(f2_13,2)))];
        f2_14 = [f2_14, abs(shift_14'.*ones(size(f2_14,1),size(f2_14,2)))];
        f2_15 = [f2_15, abs(shift_15'.*ones(size(f2_15,1),size(f2_15,2)))];
        f2_16 = [f2_16, abs(shift_16'.*ones(size(f2_16,1),size(f2_16,2)))];
        f2_17 = [f2_17, abs(shift_17'.*ones(size(f2_17,1),size(f2_17,2)))];
        f2_18 = [f2_18, abs(shift_18'.*ones(size(f2_18,1),size(f2_18,2)))];
        f2_19 = [f2_19, abs(shift_19'.*ones(size(f2_19,1),size(f2_19,2)))];
        f2_20 = [f2_20, abs(shift_20'.*ones(size(f2_20,1),size(f2_20,2)))];
        f2_21 = [f2_21, abs(shift_21'.*ones(size(f2_21,1),size(f2_21,2)))];
        f2_22 = [f2_22, abs(shift_22'.*ones(size(f2_22,1),size(f2_22,2)))];
        f2_23 = [f2_23, abs(shift_23'.*ones(size(f2_23,1),size(f2_23,2)))];
        f2_24 = [f2_24, abs(shift_24'.*ones(size(f2_24,1),size(f2_24,2)))];
        f2_25 = [f2_25, abs(shift_25'.*ones(size(f2_25,1),size(f2_25,2)))];
        f2_26 = [f2_26, abs(shift_26'.*ones(size(f2_26,1),size(f2_26,2)))];
        f2_27 = [f2_27, abs(shift_27'.*ones(size(f2_27,1),size(f2_27,2)))];
        f2_28 = [f2_28, abs(shift_28'.*ones(size(f2_28,1),size(f2_28,2)))];
        f2_29 = [f2_29, abs(shift_29'.*ones(size(f2_29,1),size(f2_29,2)))];
        f2_30 = [f2_30, abs(shift_30'.*ones(size(f2_30,1),size(f2_30,2)))];
        f2_31 = [f2_31, abs(shift_31'.*ones(size(f2_31,1),size(f2_31,2)))];
        f2_32 = [f2_32, abs(shift_32'.*ones(size(f2_32,1),size(f2_32,2)))];




        minValue = min([f2_1(:);f2_2(:);f2_3(:);f2_4(:); f2_5(:); f2_6(:); f2_7(:);...
            f2_8(:); f2_9(:); f2_10(:); f2_11(:); f2_12(:); f2_13(:); f2_14(:);...
            f2_15(:); f2_16(:); f2_17(:); f2_18(:); f2_19(:); f2_20(:); f2_21(:);...
            f2_22(:); f2_23(:); f2_24(:); f2_25(:); f2_26(:); f2_27(:); f2_28(:);...
            f2_29(:); f2_30(:); f2_31(:); f2_32(:)]);

        maxValue = max([f2_1(:);f2_2(:);f2_3(:);f2_4(:); f2_5(:); f2_6(:); f2_7(:);...
            f2_8(:); f2_9(:); f2_10(:); f2_11(:); f2_12(:); f2_13(:); f2_14(:);...
            f2_15(:); f2_16(:); f2_17(:); f2_18(:); f2_19(:); f2_20(:); f2_21(:);...
            f2_22(:); f2_23(:); f2_24(:); f2_25(:); f2_26(:); f2_27(:); f2_28(:);...
            f2_29(:); f2_30(:); f2_31(:); f2_32(:)]);

        a_h = (Gon - Goff)/(maxValue - minValue);
        b_h = Gon - a_h *(maxValue);
        G_f2_1 = a_h .* f2_1 + b_h;
        G_f2_2 = a_h .* f2_2 + b_h;
        G_f2_3 = a_h .* f2_3 + b_h;
        G_f2_4 = a_h .* f2_4 + b_h; 
        G_f2_5 = a_h .* f2_5 + b_h; 
        G_f2_6 = a_h .* f2_6 + b_h; 
        G_f2_7 = a_h .* f2_7 + b_h; 
        G_f2_8 = a_h .* f2_8 + b_h; 
        G_f2_9 = a_h .* f2_9 + b_h; 
        G_f2_10 = a_h .* f2_10 + b_h; 
        G_f2_11 = a_h .* f2_11 + b_h; 
        G_f2_12 = a_h .* f2_12 + b_h; 
        G_f2_13 = a_h .* f2_13 + b_h; 
        G_f2_14 = a_h .* f2_14 + b_h; 
        G_f2_15 = a_h .* f2_15 + b_h; 
        G_f2_16 = a_h .* f2_16 + b_h; 
        G_f2_17 = a_h .* f2_17 + b_h; 
        G_f2_18 = a_h .* f2_18 + b_h; 
        G_f2_19 = a_h .* f2_19 + b_h; 
        G_f2_20 = a_h .* f2_20 + b_h; 
        G_f2_21 = a_h .* f2_21 + b_h; 
        G_f2_22 = a_h .* f2_22 + b_h; 
        G_f2_23 = a_h .* f2_23 + b_h; 
        G_f2_24 = a_h .* f2_24 + b_h; 
        G_f2_25 = a_h .* f2_25 + b_h; 
        G_f2_26 = a_h .* f2_26 + b_h; 
        G_f2_27 = a_h .* f2_27 + b_h; 
        G_f2_28 = a_h .* f2_28 + b_h; 
        G_f2_29 = a_h .* f2_29 + b_h; 
        G_f2_30 = a_h .* f2_30 + b_h; 
        G_f2_31 = a_h .* f2_31 + b_h; 
        G_f2_32 = a_h .* f2_32 + b_h; 


        out_1_2 = im_1*f2_1;
        out_2_2 = im_2*f2_2;
        out_3_2 = im_3*f2_3;
        out_4_2 = im_4*f2_4;
        out_5_2 = im_5*f2_5;
        out_6_2 = im_6*f2_6;
        out_7_2 = im_7*f2_7;
        out_8_2 = im_8*f2_8;
        out_9_2 = im_9*f2_9;
        out_10_2 = im_10*f2_10;
        out_11_2 = im_11*f2_11;
        out_12_2 = im_12*f2_12;
        out_13_2 = im_13*f2_13;
        out_14_2 = im_14*f2_14;
        out_15_2 = im_15*f2_15;
        out_16_2 = im_16*f2_16;
        out_17_2 = im_17*f2_17;
        out_18_2 = im_18*f2_18;
        out_19_2 = im_19*f2_19;
        out_20_2 = im_20*f2_20;
        out_21_2 = im_21*f2_21;
        out_22_2 = im_22*f2_22;
        out_23_2 = im_23*f2_23;
        out_24_2 = im_24*f2_24;
        out_25_2 = im_25*f2_25;
        out_26_2 = im_26*f2_26;
        out_27_2 = im_27*f2_27;
        out_28_2 = im_28*f2_28;
        out_29_2 = im_29*f2_29;
        out_30_2 = im_30*f2_30;
        out_31_2 = im_31*f2_31;
        out_32_2 = im_32*f2_32;


        output_second_layer_with_shifting = out_1_2 + out_2_2 + out_3_2 + out_4_2 + out_5_2 +...
            out_6_2 + out_7_2 + out_8_2 + out_9_2 + out_10_2 + out_11_2 + out_12_2 +...
            out_13_2 + out_14_2 + out_15_2 + out_16_2 + out_17_2 + out_18_2 + out_19_2 +...
            out_20_2 + out_21_2 + out_22_2 + out_23_2 + out_24_2 + out_25_2 + out_26_2 +...
            out_27_2 + out_28_2 + out_29_2 + out_30_2 + out_31_2 + out_32_2;

        shifting_matrix2 = output_second_layer_with_shifting(:,33:end);
        output_second_layer_with_shifting = output_second_layer_with_shifting(:,1:32);

        % Each col is an output filter (its flattening happened row-by-row)
        output_after_shifting2 = output_second_layer_with_shifting - shifting_matrix2;

        % Re-shape it to pass it through pooling and relu
        output_2_shifting = permute(reshape(output_after_shifting2,[16, 16,32]),[2 1 3]);

        
        
            new_val_f2_1 = cell(1,1); 
            new_val_f2_2 = cell(1,1); 
            new_val_f2_3 = cell(1,1); 
            new_val_f2_4 = cell(1,1); 
            new_val_f2_5 = cell(1,1); 
            new_val_f2_6 = cell(1,1); 
            new_val_f2_7 = cell(1,1); 
            new_val_f2_8 = cell(1,1); 
            new_val_f2_9 = cell(1,1); 
            new_val_f2_10 = cell(1,1); 
            new_val_f2_11 = cell(1,1); 
            new_val_f2_12 = cell(1,1); 
            new_val_f2_13 = cell(1,1); 
            new_val_f2_14 = cell(1,1); 
            new_val_f2_15 = cell(1,1); 
            new_val_f2_16 = cell(1,1); 
            new_val_f2_17 = cell(1,1); 
            new_val_f2_18 = cell(1,1); 
            new_val_f2_19 = cell(1,1); 
            new_val_f2_20 = cell(1,1); 
            new_val_f2_21 = cell(1,1); 
            new_val_f2_22 = cell(1,1); 
            new_val_f2_23 = cell(1,1); 
            new_val_f2_24 = cell(1,1); 
            new_val_f2_25 = cell(1,1); 
            new_val_f2_26 = cell(1,1); 
            new_val_f2_27 = cell(1,1); 
            new_val_f2_28 = cell(1,1); 
            new_val_f2_29 = cell(1,1); 
            new_val_f2_30 = cell(1,1); 
            new_val_f2_31 = cell(1,1); 
            new_val_f2_32 = cell(1,1);  
 

            for i = 1: length(edges)-1
                range_mean = [edges(i) edges(i+1)];
                pd = makedist('normal','mu', mean(range_mean), 'sigma', mean(range_mean) + (variation*mean(range_mean)));
                upper = mean(range_mean) + (variation*mean(range_mean));
                lower = mean(range_mean) - (variation*mean(range_mean));
                t = truncate(pd, lower,upper);

                %Second Layer
                new_val_f2_1{i} = random(t,1,1); 
                new_val_f2_2{i} = random(t,1,1); 
                new_val_f2_3{i} = random(t,1,1); 
                new_val_f2_4{i} = random(t,1,1); 
                new_val_f2_5{i} = random(t,1,1); 
                new_val_f2_6{i} = random(t,1,1); 
                new_val_f2_7{i} = random(t,1,1); 
                new_val_f2_8{i} = random(t,1,1); 
                new_val_f2_9{i} = random(t,1,1); 
                new_val_f2_10{i} = random(t,1,1); 
                new_val_f2_11{i} = random(t,1,1); 
                new_val_f2_12{i} = random(t,1,1); 
                new_val_f2_13{i} = random(t,1,1); 
                new_val_f2_14{i} = random(t,1,1); 
                new_val_f2_15{i} = random(t,1,1); 
                new_val_f2_16{i} = random(t,1,1); 
                new_val_f2_17{i} = random(t,1,1); 
                new_val_f2_18{i} = random(t,1,1); 
                new_val_f2_19{i} = random(t,1,1); 
                new_val_f2_20{i} = random(t,1,1); 
                new_val_f2_21{i} = random(t,1,1); 
                new_val_f2_22{i} = random(t,1,1); 
                new_val_f2_23{i} = random(t,1,1); 
                new_val_f2_24{i} = random(t,1,1); 
                new_val_f2_25{i} = random(t,1,1); 
                new_val_f2_26{i} = random(t,1,1); 
                new_val_f2_27{i} = random(t,1,1); 
                new_val_f2_28{i} = random(t,1,1); 
                new_val_f2_29{i} = random(t,1,1); 
                new_val_f2_30{i} = random(t,1,1); 
                new_val_f2_31{i} = random(t,1,1); 
                new_val_f2_32{i} = random(t,1,1);

%                 new_val_f2_1{i} = mean(range_mean); 
%                 new_val_f2_2{i} = mean(range_mean); 
%                 new_val_f2_3{i} = mean(range_mean); 
%                 new_val_f2_4{i} = mean(range_mean); 
%                 new_val_f2_5{i} = mean(range_mean); 
%                 new_val_f2_6{i} = mean(range_mean); 
%                 new_val_f2_7{i} = mean(range_mean); 
%                 new_val_f2_8{i} = mean(range_mean); 
%                 new_val_f2_9{i} = mean(range_mean); 
%                 new_val_f2_10{i} = mean(range_mean); 
%                 new_val_f2_11{i} = mean(range_mean); 
%                 new_val_f2_12{i} = mean(range_mean); 
%                 new_val_f2_13{i} = mean(range_mean); 
%                 new_val_f2_14{i} = mean(range_mean); 
%                 new_val_f2_15{i} = mean(range_mean); 
%                 new_val_f2_16{i} = mean(range_mean); 
%                 new_val_f2_17{i} = mean(range_mean); 
%                 new_val_f2_18{i} = mean(range_mean); 
%                 new_val_f2_19{i} = mean(range_mean); 
%                 new_val_f2_20{i} = mean(range_mean); 
%                 new_val_f2_21{i} = mean(range_mean); 
%                 new_val_f2_22{i} = mean(range_mean); 
%                 new_val_f2_23{i} = mean(range_mean); 
%                 new_val_f2_24{i} = mean(range_mean); 
%                 new_val_f2_25{i} = mean(range_mean); 
%                 new_val_f2_26{i} = mean(range_mean); 
%                 new_val_f2_27{i} = mean(range_mean); 
%                 new_val_f2_28{i} = mean(range_mean); 
%                 new_val_f2_29{i} = mean(range_mean); 
%                 new_val_f2_30{i} = mean(range_mean); 
%                 new_val_f2_31{i} = mean(range_mean); 
%                 new_val_f2_32{i} = mean(range_mean);
             end
        
        
                    %Second Layer
            new_val_f2_1 = cell2mat(new_val_f2_1); 
            new_val_f2_2 = cell2mat(new_val_f2_2); 
            new_val_f2_3 = cell2mat(new_val_f2_3); 
            new_val_f2_4 = cell2mat(new_val_f2_4); 
            new_val_f2_5 = cell2mat(new_val_f2_5); 
            new_val_f2_6 = cell2mat(new_val_f2_6); 
            new_val_f2_7 = cell2mat(new_val_f2_7); 
            new_val_f2_8 = cell2mat(new_val_f2_8); 
            new_val_f2_9 = cell2mat(new_val_f2_9); 
            new_val_f2_10 = cell2mat(new_val_f2_10); 
            new_val_f2_11 = cell2mat(new_val_f2_11); 
            new_val_f2_12 = cell2mat(new_val_f2_12); 
            new_val_f2_13 = cell2mat(new_val_f2_13); 
            new_val_f2_14 = cell2mat(new_val_f2_14); 
            new_val_f2_15 = cell2mat(new_val_f2_15); 
            new_val_f2_16 = cell2mat(new_val_f2_16); 
            new_val_f2_17 = cell2mat(new_val_f2_17); 
            new_val_f2_18 = cell2mat(new_val_f2_18); 
            new_val_f2_19 = cell2mat(new_val_f2_19); 
            new_val_f2_20 = cell2mat(new_val_f2_20); 
            new_val_f2_21 = cell2mat(new_val_f2_21); 
            new_val_f2_22 = cell2mat(new_val_f2_22); 
            new_val_f2_23 = cell2mat(new_val_f2_23); 
            new_val_f2_24 = cell2mat(new_val_f2_24); 
            new_val_f2_25 = cell2mat(new_val_f2_25); 
            new_val_f2_26 = cell2mat(new_val_f2_26); 
            new_val_f2_27 = cell2mat(new_val_f2_27); 
            new_val_f2_28 = cell2mat(new_val_f2_28); 
            new_val_f2_29 = cell2mat(new_val_f2_29); 
            new_val_f2_30 = cell2mat(new_val_f2_30); 
            new_val_f2_31 = cell2mat(new_val_f2_31); 
            new_val_f2_32 = cell2mat(new_val_f2_32); 
        
        
            G_f2_1_q = zeros(size(G_f2_1,1),size(G_f2_1,2)); 
            G_f2_2_q = zeros(size(G_f2_2,1),size(G_f2_2,2)); 
            G_f2_3_q = zeros(size(G_f2_3,1),size(G_f2_3,2)); 
            G_f2_4_q = zeros(size(G_f2_4,1),size(G_f2_4,2)); 
            G_f2_5_q = zeros(size(G_f2_5,1),size(G_f2_5,2)); 
            G_f2_6_q = zeros(size(G_f2_6,1),size(G_f2_6,2)); 
            G_f2_7_q = zeros(size(G_f2_7,1),size(G_f2_7,2)); 
            G_f2_8_q = zeros(size(G_f2_8,1),size(G_f2_8,2)); 
            G_f2_9_q = zeros(size(G_f2_9,1),size(G_f2_9,2)); 
            G_f2_10_q = zeros(size(G_f2_10,1),size(G_f2_10,2)); 
            G_f2_11_q = zeros(size(G_f2_11,1),size(G_f2_11,2)); 
            G_f2_12_q = zeros(size(G_f2_12,1),size(G_f2_12,2)); 
            G_f2_13_q = zeros(size(G_f2_13,1),size(G_f2_13,2)); 
            G_f2_14_q = zeros(size(G_f2_14,1),size(G_f2_14,2)); 
            G_f2_15_q = zeros(size(G_f2_15,1),size(G_f2_15,2)); 
            G_f2_16_q = zeros(size(G_f2_16,1),size(G_f2_16,2)); 
            G_f2_17_q = zeros(size(G_f2_17,1),size(G_f2_17,2)); 
            G_f2_18_q = zeros(size(G_f2_18,1),size(G_f2_18,2)); 
            G_f2_19_q = zeros(size(G_f2_19,1),size(G_f2_19,2)); 
            G_f2_20_q = zeros(size(G_f2_20,1),size(G_f2_20,2)); 
            G_f2_21_q = zeros(size(G_f2_21,1),size(G_f2_21,2)); 
            G_f2_22_q = zeros(size(G_f2_22,1),size(G_f2_22,2)); 
            G_f2_23_q = zeros(size(G_f2_23,1),size(G_f2_23,2)); 
            G_f2_24_q = zeros(size(G_f2_24,1),size(G_f2_24,2)); 
            G_f2_25_q = zeros(size(G_f2_25,1),size(G_f2_25,2)); 
            G_f2_26_q = zeros(size(G_f2_26,1),size(G_f2_26,2)); 
            G_f2_27_q = zeros(size(G_f2_27,1),size(G_f2_27,2)); 
            G_f2_28_q = zeros(size(G_f2_28,1),size(G_f2_28,2)); 
            G_f2_29_q = zeros(size(G_f2_29,1),size(G_f2_29,2)); 
            G_f2_30_q = zeros(size(G_f2_30,1),size(G_f2_30,2)); 
            G_f2_31_q = zeros(size(G_f2_31,1),size(G_f2_31,2)); 
            G_f2_32_q = zeros(size(G_f2_32,1),size(G_f2_32,2)); 

        
              for i = 1:size(G_f2_1,1)
                for j = 1:size(G_f2_1,2)
                    if G_f2_1(i,j) < edges(1) 
                        G_f2_1_q(i,j) = new_val_f2_1(1);

                    elseif G_f2_1(i,j) >= edges(1) && G_f2_1(i,j) < edges(2)
                        G_f2_1_q(i,j) = new_val_f2_1(1);

                    elseif G_f2_1(i,j) >= edges(2) && G_f2_1(i,j) < edges(3)
                        G_f2_1_q(i,j) = new_val_f2_1(2);   

                    elseif G_f2_1(i,j) >= edges(3) && G_f2_1(i,j) < edges(4)
                        G_f2_1_q(i,j) = new_val_f2_1(3);

                    elseif G_f2_1(i,j) >= edges(4) && G_f2_1(i,j) < edges(5)
                        G_f2_1_q(i,j) = new_val_f2_1(4); 

                    elseif G_f2_1(i,j) >= edges(5) && G_f2_1(i,j) < edges(6)
                        G_f2_1_q(i,j) = new_val_f2_1(5);          

                    elseif G_f2_1(i,j) >= edges(6) && G_f2_1(i,j) < edges(7)
                        G_f2_1_q(i,j) = new_val_f2_1(6);

                    elseif G_f2_1(i,j) >= edges(7) && G_f2_1(i,j) < edges(8)
                        G_f2_1_q(i,j) = new_val_f2_1(7); 

                    elseif G_f2_1(i,j) >= edges(8) && G_f2_1(i,j) < edges(9)
                        G_f2_1_q(i,j) = new_val_f2_1(8);          

                    elseif G_f2_1(i,j) >= edges(9) && G_f2_1(i,j) < edges(10)
                        G_f2_1_q(i,j) = new_val_f2_1(9);

                    elseif G_f2_1(i,j) >= edges(10) && G_f2_1(i,j) < edges(11)
                        G_f2_1_q(i,j) = new_val_f2_1(10);

                    elseif G_f2_1(i,j) >= edges(11) && G_f2_1(i,j) < edges(12)
                        G_f2_1_q(i,j) = new_val_f2_1(11);

                    elseif G_f2_1(i,j) >= edges(12) && G_f2_1(i,j) < edges(13)
                        G_f2_1_q(i,j) = new_val_f2_1(12);

                    elseif G_f2_1(i,j) >= edges(13) && G_f2_1(i,j) < edges(14) 
                         G_f2_1_q(i,j) = new_val_f2_1(13);

                    elseif G_f2_1(i,j) >= edges(14) && G_f2_1(i,j) < edges(15) 
                        G_f2_1_q(i,j) = new_val_f2_1(14);

                    elseif G_f2_1(i,j) >= edges(15) && G_f2_1(i,j) < edges(16)
                        G_f2_1_q(i,j) = new_val_f2_1(15);

                    elseif G_f2_1(i,j) >= edges(16) && G_f2_1(i,j) <= edges(17) 
                         G_f2_1_q(i,j) = new_val_f2_1(16);            
                    end
                end
             end   

            for i = 1:size(G_f2_2,1)
                for j = 1:size(G_f2_2,2)
                    if G_f2_2(i,j) < edges(1) 
                        G_f2_2_q(i,j) = new_val_f2_2(1);

                    elseif G_f2_2(i,j) >= edges(1) && G_f2_2(i,j) < edges(2)
                        G_f2_2_q(i,j) = new_val_f2_2(1);

                    elseif G_f2_2(i,j) >= edges(2) && G_f2_2(i,j) < edges(3)
                        G_f2_2_q(i,j) = new_val_f2_2(2);   

                    elseif G_f2_2(i,j) >= edges(3) && G_f2_2(i,j) < edges(4)
                        G_f2_2_q(i,j) = new_val_f2_2(3);

                    elseif G_f2_2(i,j) >= edges(4) && G_f2_2(i,j) < edges(5)
                        G_f2_2_q(i,j) = new_val_f2_2(4); 

                    elseif G_f2_2(i,j) >= edges(5) && G_f2_2(i,j) < edges(6)
                        G_f2_2_q(i,j) = new_val_f2_2(5);          

                    elseif G_f2_2(i,j) >= edges(6) && G_f2_2(i,j) < edges(7)
                        G_f2_2_q(i,j) = new_val_f2_2(6);

                    elseif G_f2_2(i,j) >= edges(7) && G_f2_2(i,j) < edges(8)
                        G_f2_2_q(i,j) = new_val_f2_2(7); 

                    elseif G_f2_2(i,j) >= edges(8) && G_f2_2(i,j) < edges(9)
                        G_f2_2_q(i,j) = new_val_f2_2(8);          

                    elseif G_f2_2(i,j) >= edges(9) && G_f2_2(i,j) < edges(10)
                        G_f2_2_q(i,j) = new_val_f2_2(9);

                    elseif G_f2_2(i,j) >= edges(10) && G_f2_2(i,j) < edges(11)
                        G_f2_2_q(i,j) = new_val_f2_2(10);

                    elseif G_f2_2(i,j) >= edges(11) && G_f2_2(i,j) < edges(12)
                        G_f2_2_q(i,j) = new_val_f2_2(11);

                    elseif G_f2_2(i,j) >= edges(12) && G_f2_2(i,j) < edges(13)
                        G_f2_2_q(i,j) = new_val_f2_2(12);

                    elseif G_f2_2(i,j) >= edges(13) && G_f2_2(i,j) < edges(14) 
                         G_f2_2_q(i,j) = new_val_f2_2(13);

                    elseif G_f2_2(i,j) >= edges(14) && G_f2_2(i,j) < edges(15) 
                        G_f2_2_q(i,j) = new_val_f2_2(14);

                    elseif G_f2_2(i,j) >= edges(15) && G_f2_2(i,j) < edges(16)
                        G_f2_2_q(i,j) = new_val_f2_2(15);

                    elseif G_f2_2(i,j) >= edges(16) && G_f2_2(i,j) <= edges(17) 
                         G_f2_2_q(i,j) = new_val_f2_2(16);            
                    end
                end
            end      

                for i = 1:size(G_f2_3,1)
                for j = 1:size(G_f2_3,2)
                    if G_f2_3(i,j) < edges(1) 
                        G_f2_3_q(i,j) = new_val_f2_3(1);

                    elseif G_f2_3(i,j) >= edges(1) && G_f2_3(i,j) < edges(2)
                        G_f2_3_q(i,j) = new_val_f2_3(1);

                    elseif G_f2_3(i,j) >= edges(2) && G_f2_3(i,j) < edges(3)
                        G_f2_3_q(i,j) = new_val_f2_3(2);   

                    elseif G_f2_3(i,j) >= edges(3) && G_f2_3(i,j) < edges(4)
                        G_f2_3_q(i,j) = new_val_f2_3(3);

                    elseif G_f2_3(i,j) >= edges(4) && G_f2_3(i,j) < edges(5)
                        G_f2_3_q(i,j) = new_val_f2_3(4); 

                    elseif G_f2_3(i,j) >= edges(5) && G_f2_3(i,j) < edges(6)
                        G_f2_3_q(i,j) = new_val_f2_3(5);          

                    elseif G_f2_3(i,j) >= edges(6) && G_f2_3(i,j) < edges(7)
                        G_f2_3_q(i,j) = new_val_f2_3(6);

                    elseif G_f2_3(i,j) >= edges(7) && G_f2_3(i,j) < edges(8)
                        G_f2_3_q(i,j) = new_val_f2_3(7); 

                    elseif G_f2_3(i,j) >= edges(8) && G_f2_3(i,j) < edges(9)
                        G_f2_3_q(i,j) = new_val_f2_3(8);          

                    elseif G_f2_3(i,j) >= edges(9) && G_f2_3(i,j) < edges(10)
                        G_f2_3_q(i,j) = new_val_f2_3(9);

                    elseif G_f2_3(i,j) >= edges(10) && G_f2_3(i,j) < edges(11)
                        G_f2_3_q(i,j) = new_val_f2_3(10);

                    elseif G_f2_3(i,j) >= edges(11) && G_f2_3(i,j) < edges(12)
                        G_f2_3_q(i,j) = new_val_f2_3(11);

                    elseif G_f2_3(i,j) >= edges(12) && G_f2_3(i,j) < edges(13)
                        G_f2_3_q(i,j) = new_val_f2_3(12);

                    elseif G_f2_3(i,j) >= edges(13) && G_f2_3(i,j) < edges(14) 
                         G_f2_3_q(i,j) = new_val_f2_3(13);

                    elseif G_f2_3(i,j) >= edges(14) && G_f2_3(i,j) < edges(15) 
                        G_f2_3_q(i,j) = new_val_f2_3(14);

                    elseif G_f2_3(i,j) >= edges(15) && G_f2_3(i,j) < edges(16)
                        G_f2_3_q(i,j) = new_val_f2_3(15);

                    elseif G_f2_3(i,j) >= edges(16) && G_f2_3(i,j) <= edges(17) 
                         G_f2_3_q(i,j) = new_val_f2_3(16);            
                    end
                end
                end  

            for i = 1:size(G_f2_4,1)
                for j = 1:size(G_f2_4,2)
                    if G_f2_4(i,j) < edges(1) 
                        G_f2_4_q(i,j) = new_val_f2_4(1);

                    elseif G_f2_4(i,j) >= edges(1) && G_f2_4(i,j) < edges(2)
                        G_f2_4_q(i,j) = new_val_f2_4(1);

                    elseif G_f2_4(i,j) >= edges(2) && G_f2_4(i,j) < edges(3)
                        G_f2_4_q(i,j) = new_val_f2_4(2);   

                    elseif G_f2_4(i,j) >= edges(3) && G_f2_4(i,j) < edges(4)
                        G_f2_4_q(i,j) = new_val_f2_4(3);

                    elseif G_f2_4(i,j) >= edges(4) && G_f2_4(i,j) < edges(5)
                        G_f2_4_q(i,j) = new_val_f2_4(4); 

                    elseif G_f2_4(i,j) >= edges(5) && G_f2_4(i,j) < edges(6)
                        G_f2_4_q(i,j) = new_val_f2_4(5);          

                    elseif G_f2_4(i,j) >= edges(6) && G_f2_4(i,j) < edges(7)
                        G_f2_4_q(i,j) = new_val_f2_4(6);

                    elseif G_f2_4(i,j) >= edges(7) && G_f2_4(i,j) < edges(8)
                        G_f2_4_q(i,j) = new_val_f2_4(7); 

                    elseif G_f2_4(i,j) >= edges(8) && G_f2_4(i,j) < edges(9)
                        G_f2_4_q(i,j) = new_val_f2_4(8);          

                    elseif G_f2_4(i,j) >= edges(9) && G_f2_4(i,j) < edges(10)
                        G_f2_4_q(i,j) = new_val_f2_4(9);

                    elseif G_f2_4(i,j) >= edges(10) && G_f2_4(i,j) < edges(11)
                        G_f2_4_q(i,j) = new_val_f2_4(10);

                    elseif G_f2_4(i,j) >= edges(11) && G_f2_4(i,j) < edges(12)
                        G_f2_4_q(i,j) = new_val_f2_4(11);

                    elseif G_f2_4(i,j) >= edges(12) && G_f2_4(i,j) < edges(13)
                        G_f2_4_q(i,j) = new_val_f2_4(12);

                    elseif G_f2_4(i,j) >= edges(13) && G_f2_4(i,j) < edges(14) 
                         G_f2_4_q(i,j) = new_val_f2_4(13);

                    elseif G_f2_4(i,j) >= edges(14) && G_f2_4(i,j) < edges(15) 
                        G_f2_4_q(i,j) = new_val_f2_4(14);

                    elseif G_f2_4(i,j) >= edges(15) && G_f2_4(i,j) < edges(16)
                        G_f2_4_q(i,j) = new_val_f2_4(15);

                    elseif G_f2_4(i,j) >= edges(16) && G_f2_4(i,j) <= edges(17) 
                         G_f2_4_q(i,j) = new_val_f2_4(16);            
                    end
                end
            end  

            for i = 1:size(G_f2_5,1)
                for j = 1:size(G_f2_5,2)
                    if G_f2_5(i,j) < edges(1) 
                        G_f2_5_q(i,j) = new_val_f2_5(1);

                    elseif G_f2_5(i,j) >= edges(1) && G_f2_5(i,j) < edges(2)
                        G_f2_5_q(i,j) = new_val_f2_5(1);

                    elseif G_f2_5(i,j) >= edges(2) && G_f2_5(i,j) < edges(3)
                        G_f2_5_q(i,j) = new_val_f2_5(2);   

                    elseif G_f2_5(i,j) >= edges(3) && G_f2_5(i,j) < edges(4)
                        G_f2_5_q(i,j) = new_val_f2_5(3);

                    elseif G_f2_5(i,j) >= edges(4) && G_f2_5(i,j) < edges(5)
                        G_f2_5_q(i,j) = new_val_f2_5(4); 

                    elseif G_f2_5(i,j) >= edges(5) && G_f2_5(i,j) < edges(6)
                        G_f2_5_q(i,j) = new_val_f2_5(5);          

                    elseif G_f2_5(i,j) >= edges(6) && G_f2_5(i,j) < edges(7)
                        G_f2_5_q(i,j) = new_val_f2_5(6);

                    elseif G_f2_5(i,j) >= edges(7) && G_f2_5(i,j) < edges(8)
                        G_f2_5_q(i,j) = new_val_f2_5(7); 

                    elseif G_f2_5(i,j) >= edges(8) && G_f2_5(i,j) < edges(9)
                        G_f2_5_q(i,j) = new_val_f2_5(8);          

                    elseif G_f2_5(i,j) >= edges(9) && G_f2_5(i,j) < edges(10)
                        G_f2_5_q(i,j) = new_val_f2_5(9);

                    elseif G_f2_5(i,j) >= edges(10) && G_f2_5(i,j) < edges(11)
                        G_f2_5_q(i,j) = new_val_f2_5(10);

                    elseif G_f2_5(i,j) >= edges(11) && G_f2_5(i,j) < edges(12)
                        G_f2_5_q(i,j) = new_val_f2_5(11);

                    elseif G_f2_5(i,j) >= edges(12) && G_f2_5(i,j) < edges(13)
                        G_f2_5_q(i,j) = new_val_f2_5(12);

                    elseif G_f2_5(i,j) >= edges(13) && G_f2_5(i,j) < edges(14) 
                         G_f2_5_q(i,j) = new_val_f2_5(13);

                    elseif G_f2_5(i,j) >= edges(14) && G_f2_5(i,j) < edges(15) 
                        G_f2_5_q(i,j) = new_val_f2_5(14);

                    elseif G_f2_5(i,j) >= edges(15) && G_f2_5(i,j) < edges(16)
                        G_f2_5_q(i,j) = new_val_f2_5(15);

                    elseif G_f2_5(i,j) >= edges(16) && G_f2_5(i,j) <= edges(17) 
                         G_f2_5_q(i,j) = new_val_f2_5(16);            
                    end
                end
            end      

            for i = 1:size(G_f2_6,1)
                for j = 1:size(G_f2_6,2)
                    if G_f2_6(i,j) < edges(1) 
                        G_f2_6_q(i,j) = new_val_f2_6(1);

                    elseif G_f2_6(i,j) >= edges(1) && G_f2_6(i,j) < edges(2)
                        G_f2_6_q(i,j) = new_val_f2_6(1);

                    elseif G_f2_6(i,j) >= edges(2) && G_f2_6(i,j) < edges(3)
                        G_f2_6_q(i,j) = new_val_f2_6(2);   

                    elseif G_f2_6(i,j) >= edges(3) && G_f2_6(i,j) < edges(4)
                        G_f2_6_q(i,j) = new_val_f2_6(3);

                    elseif G_f2_6(i,j) >= edges(4) && G_f2_6(i,j) < edges(5)
                        G_f2_6_q(i,j) = new_val_f2_6(4); 

                    elseif G_f2_6(i,j) >= edges(5) && G_f2_6(i,j) < edges(6)
                        G_f2_6_q(i,j) = new_val_f2_6(5);          

                    elseif G_f2_6(i,j) >= edges(6) && G_f2_6(i,j) < edges(7)
                        G_f2_6_q(i,j) = new_val_f2_6(6);

                    elseif G_f2_6(i,j) >= edges(7) && G_f2_6(i,j) < edges(8)
                        G_f2_6_q(i,j) = new_val_f2_6(7); 

                    elseif G_f2_6(i,j) >= edges(8) && G_f2_6(i,j) < edges(9)
                        G_f2_6_q(i,j) = new_val_f2_6(8);          

                    elseif G_f2_6(i,j) >= edges(9) && G_f2_6(i,j) < edges(10)
                        G_f2_6_q(i,j) = new_val_f2_6(9);

                    elseif G_f2_6(i,j) >= edges(10) && G_f2_6(i,j) < edges(11)
                        G_f2_6_q(i,j) = new_val_f2_6(10);

                    elseif G_f2_6(i,j) >= edges(11) && G_f2_6(i,j) < edges(12)
                        G_f2_6_q(i,j) = new_val_f2_6(11);

                    elseif G_f2_6(i,j) >= edges(12) && G_f2_6(i,j) < edges(13)
                        G_f2_6_q(i,j) = new_val_f2_6(12);

                    elseif G_f2_6(i,j) >= edges(13) && G_f2_6(i,j) < edges(14) 
                         G_f2_6_q(i,j) = new_val_f2_6(13);

                    elseif G_f2_6(i,j) >= edges(14) && G_f2_6(i,j) < edges(15) 
                        G_f2_6_q(i,j) = new_val_f2_6(14);

                    elseif G_f2_6(i,j) >= edges(15) && G_f2_6(i,j) < edges(16)
                        G_f2_6_q(i,j) = new_val_f2_6(15);

                    elseif G_f2_6(i,j) >= edges(16) && G_f2_6(i,j) <= edges(17) 
                         G_f2_6_q(i,j) = new_val_f2_6(16);            
                    end
                end
            end      

                for i = 1:size(G_f2_7,1) 
                for j = 1:size(G_f2_7,2) 
                    if G_f2_7(i,j) < edges(1)  
                        G_f2_7_q(i,j) = new_val_f2_7(1);  
                    elseif G_f2_7(i,j) >= edges(1) && G_f2_7(i,j) < edges(2)  
                        G_f2_7_q(i,j) = new_val_f2_7(1);  
                    elseif G_f2_7(i,j) >= edges(2) && G_f2_7(i,j) < edges(3)  
                        G_f2_7_q(i,j) = new_val_f2_7(2);    
                    elseif G_f2_7(i,j) >= edges(3) && G_f2_7(i,j) < edges(4)  
                        G_f2_7_q(i,j) = new_val_f2_7(3);  
                    elseif G_f2_7(i,j) >= edges(4) && G_f2_7(i,j) < edges(5)  
                        G_f2_7_q(i,j) = new_val_f2_7(4);   
                    elseif G_f2_7(i,j) >= edges(5) && G_f2_7(i,j) < edges(6)  
                        G_f2_7_q(i,j) = new_val_f2_7(5);     
                    elseif G_f2_7(i,j) >= edges(6) && G_f2_7(i,j) < edges(7)  
                        G_f2_7_q(i,j) = new_val_f2_7(6);  
                    elseif G_f2_7(i,j) >= edges(7) && G_f2_7(i,j) < edges(8)  
                        G_f2_7_q(i,j) = new_val_f2_7(7);   
                    elseif G_f2_7(i,j) >= edges(8) && G_f2_7(i,j) < edges(9)  
                        G_f2_7_q(i,j) = new_val_f2_7(8);     
                    elseif G_f2_7(i,j) >= edges(9) && G_f2_7(i,j) < edges(10)  
                        G_f2_7_q(i,j) = new_val_f2_7(9);  
                    elseif G_f2_7(i,j) >= edges(10) && G_f2_7(i,j) < edges(11)  
                        G_f2_7_q(i,j) = new_val_f2_7(10);  
                    elseif G_f2_7(i,j) >= edges(11) && G_f2_7(i,j) < edges(12)  
                        G_f2_7_q(i,j) = new_val_f2_7(11);  
                    elseif G_f2_7(i,j) >= edges(12) && G_f2_7(i,j) < edges(13)  
                        G_f2_7_q(i,j) = new_val_f2_7(12);  
                    elseif G_f2_7(i,j) >= edges(13) && G_f2_7(i,j) < edges(14)   
                         G_f2_7_q(i,j) = new_val_f2_7(13);   
                    elseif G_f2_7(i,j) >= edges(14) && G_f2_7(i,j) < edges(15)  
                        G_f2_7_q(i,j) = new_val_f2_7(14);  
                    elseif G_f2_7(i,j) >= edges(15) && G_f2_7(i,j) < edges(16)  
                        G_f2_7_q(i,j) = new_val_f2_7(15);  
                    elseif G_f2_7(i,j) >= edges(16) && G_f2_7(i,j) <= edges(17)   
                         G_f2_7_q(i,j) = new_val_f2_7(16);    
                    end 
                end 
                end  


            for i = 1:size(G_f2_8,1) 
                for j = 1:size(G_f2_8,2) 
                    if G_f2_8(i,j) < edges(1)  
                        G_f2_8_q(i,j) = new_val_f2_8(1);  
                    elseif G_f2_8(i,j) >= edges(1) && G_f2_8(i,j) < edges(2)  
                        G_f2_8_q(i,j) = new_val_f2_8(1);  
                    elseif G_f2_8(i,j) >= edges(2) && G_f2_8(i,j) < edges(3)  
                        G_f2_8_q(i,j) = new_val_f2_8(2);    
                    elseif G_f2_8(i,j) >= edges(3) && G_f2_8(i,j) < edges(4)  
                        G_f2_8_q(i,j) = new_val_f2_8(3);  
                    elseif G_f2_8(i,j) >= edges(4) && G_f2_8(i,j) < edges(5)  
                        G_f2_8_q(i,j) = new_val_f2_8(4);   
                    elseif G_f2_8(i,j) >= edges(5) && G_f2_8(i,j) < edges(6)  
                        G_f2_8_q(i,j) = new_val_f2_8(5);     
                    elseif G_f2_8(i,j) >= edges(6) && G_f2_8(i,j) < edges(7)  
                        G_f2_8_q(i,j) = new_val_f2_8(6);  
                    elseif G_f2_8(i,j) >= edges(7) && G_f2_8(i,j) < edges(8)  
                        G_f2_8_q(i,j) = new_val_f2_8(7);   
                    elseif G_f2_8(i,j) >= edges(8) && G_f2_8(i,j) < edges(9)  
                        G_f2_8_q(i,j) = new_val_f2_8(8);     
                    elseif G_f2_8(i,j) >= edges(9) && G_f2_8(i,j) < edges(10)  
                        G_f2_8_q(i,j) = new_val_f2_8(9);  
                    elseif G_f2_8(i,j) >= edges(10) && G_f2_8(i,j) < edges(11)  
                        G_f2_8_q(i,j) = new_val_f2_8(10);  
                    elseif G_f2_8(i,j) >= edges(11) && G_f2_8(i,j) < edges(12)  
                        G_f2_8_q(i,j) = new_val_f2_8(11);  
                    elseif G_f2_8(i,j) >= edges(12) && G_f2_8(i,j) < edges(13)  
                        G_f2_8_q(i,j) = new_val_f2_8(12);  
                    elseif G_f2_8(i,j) >= edges(13) && G_f2_8(i,j) < edges(14)   
                         G_f2_8_q(i,j) = new_val_f2_8(13);   
                    elseif G_f2_8(i,j) >= edges(14) && G_f2_8(i,j) < edges(15)  
                        G_f2_8_q(i,j) = new_val_f2_8(14);  
                    elseif G_f2_8(i,j) >= edges(15) && G_f2_8(i,j) < edges(16)  
                        G_f2_8_q(i,j) = new_val_f2_8(15);  
                    elseif G_f2_8(i,j) >= edges(16) && G_f2_8(i,j) <= edges(17)   
                         G_f2_8_q(i,j) = new_val_f2_8(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_9,1) 
                for j = 1:size(G_f2_9,2) 
                    if G_f2_9(i,j) < edges(1)  
                        G_f2_9_q(i,j) = new_val_f2_9(1);  
                    elseif G_f2_9(i,j) >= edges(1) && G_f2_9(i,j) < edges(2)  
                        G_f2_9_q(i,j) = new_val_f2_9(1);  
                    elseif G_f2_9(i,j) >= edges(2) && G_f2_9(i,j) < edges(3)  
                        G_f2_9_q(i,j) = new_val_f2_9(2);    
                    elseif G_f2_9(i,j) >= edges(3) && G_f2_9(i,j) < edges(4)  
                        G_f2_9_q(i,j) = new_val_f2_9(3);  
                    elseif G_f2_9(i,j) >= edges(4) && G_f2_9(i,j) < edges(5)  
                        G_f2_9_q(i,j) = new_val_f2_9(4);   
                    elseif G_f2_9(i,j) >= edges(5) && G_f2_9(i,j) < edges(6)  
                        G_f2_9_q(i,j) = new_val_f2_9(5);     
                    elseif G_f2_9(i,j) >= edges(6) && G_f2_9(i,j) < edges(7)  
                        G_f2_9_q(i,j) = new_val_f2_9(6);  
                    elseif G_f2_9(i,j) >= edges(7) && G_f2_9(i,j) < edges(8)  
                        G_f2_9_q(i,j) = new_val_f2_9(7);   
                    elseif G_f2_9(i,j) >= edges(8) && G_f2_9(i,j) < edges(9)  
                        G_f2_9_q(i,j) = new_val_f2_9(8);     
                    elseif G_f2_9(i,j) >= edges(9) && G_f2_9(i,j) < edges(10)  
                        G_f2_9_q(i,j) = new_val_f2_9(9);  
                    elseif G_f2_9(i,j) >= edges(10) && G_f2_9(i,j) < edges(11)  
                        G_f2_9_q(i,j) = new_val_f2_9(10);  
                    elseif G_f2_9(i,j) >= edges(11) && G_f2_9(i,j) < edges(12)  
                        G_f2_9_q(i,j) = new_val_f2_9(11);  
                    elseif G_f2_9(i,j) >= edges(12) && G_f2_9(i,j) < edges(13)  
                        G_f2_9_q(i,j) = new_val_f2_9(12);  
                    elseif G_f2_9(i,j) >= edges(13) && G_f2_9(i,j) < edges(14)   
                         G_f2_9_q(i,j) = new_val_f2_9(13);   
                    elseif G_f2_9(i,j) >= edges(14) && G_f2_9(i,j) < edges(15)  
                        G_f2_9_q(i,j) = new_val_f2_9(14);  
                    elseif G_f2_9(i,j) >= edges(15) && G_f2_9(i,j) < edges(16)  
                        G_f2_9_q(i,j) = new_val_f2_9(15);  
                    elseif G_f2_9(i,j) >= edges(16) && G_f2_9(i,j) <= edges(17)   
                         G_f2_9_q(i,j) = new_val_f2_9(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_10,1) 
                for j = 1:size(G_f2_10,2) 
                    if G_f2_10(i,j) < edges(1)  
                        G_f2_10_q(i,j) = new_val_f2_10(1);  
                    elseif G_f2_10(i,j) >= edges(1) && G_f2_10(i,j) < edges(2)  
                        G_f2_10_q(i,j) = new_val_f2_10(1);  
                    elseif G_f2_10(i,j) >= edges(2) && G_f2_10(i,j) < edges(3)  
                        G_f2_10_q(i,j) = new_val_f2_10(2);    
                    elseif G_f2_10(i,j) >= edges(3) && G_f2_10(i,j) < edges(4)  
                        G_f2_10_q(i,j) = new_val_f2_10(3);  
                    elseif G_f2_10(i,j) >= edges(4) && G_f2_10(i,j) < edges(5)  
                        G_f2_10_q(i,j) = new_val_f2_10(4);   
                    elseif G_f2_10(i,j) >= edges(5) && G_f2_10(i,j) < edges(6)  
                        G_f2_10_q(i,j) = new_val_f2_10(5);     
                    elseif G_f2_10(i,j) >= edges(6) && G_f2_10(i,j) < edges(7)  
                        G_f2_10_q(i,j) = new_val_f2_10(6);  
                    elseif G_f2_10(i,j) >= edges(7) && G_f2_10(i,j) < edges(8)  
                        G_f2_10_q(i,j) = new_val_f2_10(7);   
                    elseif G_f2_10(i,j) >= edges(8) && G_f2_10(i,j) < edges(9)  
                        G_f2_10_q(i,j) = new_val_f2_10(8);     
                    elseif G_f2_10(i,j) >= edges(9) && G_f2_10(i,j) < edges(10)  
                        G_f2_10_q(i,j) = new_val_f2_10(9);  
                    elseif G_f2_10(i,j) >= edges(10) && G_f2_10(i,j) < edges(11)  
                        G_f2_10_q(i,j) = new_val_f2_10(10);  
                    elseif G_f2_10(i,j) >= edges(11) && G_f2_10(i,j) < edges(12)  
                        G_f2_10_q(i,j) = new_val_f2_10(11);  
                    elseif G_f2_10(i,j) >= edges(12) && G_f2_10(i,j) < edges(13)  
                        G_f2_10_q(i,j) = new_val_f2_10(12);  
                    elseif G_f2_10(i,j) >= edges(13) && G_f2_10(i,j) < edges(14)   
                         G_f2_10_q(i,j) = new_val_f2_10(13);   
                    elseif G_f2_10(i,j) >= edges(14) && G_f2_10(i,j) < edges(15)  
                        G_f2_10_q(i,j) = new_val_f2_10(14);  
                    elseif G_f2_10(i,j) >= edges(15) && G_f2_10(i,j) < edges(16)  
                        G_f2_10_q(i,j) = new_val_f2_10(15);  
                    elseif G_f2_10(i,j) >= edges(16) && G_f2_10(i,j) <= edges(17)   
                         G_f2_10_q(i,j) = new_val_f2_10(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_11,1) 
                for j = 1:size(G_f2_11,2) 
                    if G_f2_11(i,j) < edges(1)  
                        G_f2_11_q(i,j) = new_val_f2_11(1);  
                    elseif G_f2_11(i,j) >= edges(1) && G_f2_11(i,j) < edges(2)  
                        G_f2_11_q(i,j) = new_val_f2_11(1);  
                    elseif G_f2_11(i,j) >= edges(2) && G_f2_11(i,j) < edges(3)  
                        G_f2_11_q(i,j) = new_val_f2_11(2);    
                    elseif G_f2_11(i,j) >= edges(3) && G_f2_11(i,j) < edges(4)  
                        G_f2_11_q(i,j) = new_val_f2_11(3);  
                    elseif G_f2_11(i,j) >= edges(4) && G_f2_11(i,j) < edges(5)  
                        G_f2_11_q(i,j) = new_val_f2_11(4);   
                    elseif G_f2_11(i,j) >= edges(5) && G_f2_11(i,j) < edges(6)  
                        G_f2_11_q(i,j) = new_val_f2_11(5);     
                    elseif G_f2_11(i,j) >= edges(6) && G_f2_11(i,j) < edges(7)  
                        G_f2_11_q(i,j) = new_val_f2_11(6);  
                    elseif G_f2_11(i,j) >= edges(7) && G_f2_11(i,j) < edges(8)  
                        G_f2_11_q(i,j) = new_val_f2_11(7);   
                    elseif G_f2_11(i,j) >= edges(8) && G_f2_11(i,j) < edges(9)  
                        G_f2_11_q(i,j) = new_val_f2_11(8);     
                    elseif G_f2_11(i,j) >= edges(9) && G_f2_11(i,j) < edges(10)  
                        G_f2_11_q(i,j) = new_val_f2_11(9);  
                    elseif G_f2_11(i,j) >= edges(10) && G_f2_11(i,j) < edges(11)  
                        G_f2_11_q(i,j) = new_val_f2_11(10);  
                    elseif G_f2_11(i,j) >= edges(11) && G_f2_11(i,j) < edges(12)  
                        G_f2_11_q(i,j) = new_val_f2_11(11);  
                    elseif G_f2_11(i,j) >= edges(12) && G_f2_11(i,j) < edges(13)  
                        G_f2_11_q(i,j) = new_val_f2_11(12);  
                    elseif G_f2_11(i,j) >= edges(13) && G_f2_11(i,j) < edges(14)   
                         G_f2_11_q(i,j) = new_val_f2_11(13);   
                    elseif G_f2_11(i,j) >= edges(14) && G_f2_11(i,j) < edges(15)  
                        G_f2_11_q(i,j) = new_val_f2_11(14);  
                    elseif G_f2_11(i,j) >= edges(15) && G_f2_11(i,j) < edges(16)  
                        G_f2_11_q(i,j) = new_val_f2_11(15);  
                    elseif G_f2_11(i,j) >= edges(16) && G_f2_11(i,j) <= edges(17)   
                         G_f2_11_q(i,j) = new_val_f2_11(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_12,1) 
                for j = 1:size(G_f2_12,2) 
                    if G_f2_12(i,j) < edges(1)  
                        G_f2_12_q(i,j) = new_val_f2_12(1);  
                    elseif G_f2_12(i,j) >= edges(1) && G_f2_12(i,j) < edges(2)  
                        G_f2_12_q(i,j) = new_val_f2_12(1);  
                    elseif G_f2_12(i,j) >= edges(2) && G_f2_12(i,j) < edges(3)  
                        G_f2_12_q(i,j) = new_val_f2_12(2);    
                    elseif G_f2_12(i,j) >= edges(3) && G_f2_12(i,j) < edges(4)  
                        G_f2_12_q(i,j) = new_val_f2_12(3);  
                    elseif G_f2_12(i,j) >= edges(4) && G_f2_12(i,j) < edges(5)  
                        G_f2_12_q(i,j) = new_val_f2_12(4);   
                    elseif G_f2_12(i,j) >= edges(5) && G_f2_12(i,j) < edges(6)  
                        G_f2_12_q(i,j) = new_val_f2_12(5);     
                    elseif G_f2_12(i,j) >= edges(6) && G_f2_12(i,j) < edges(7)  
                        G_f2_12_q(i,j) = new_val_f2_12(6);  
                    elseif G_f2_12(i,j) >= edges(7) && G_f2_12(i,j) < edges(8)  
                        G_f2_12_q(i,j) = new_val_f2_12(7);   
                    elseif G_f2_12(i,j) >= edges(8) && G_f2_12(i,j) < edges(9)  
                        G_f2_12_q(i,j) = new_val_f2_12(8);     
                    elseif G_f2_12(i,j) >= edges(9) && G_f2_12(i,j) < edges(10)  
                        G_f2_12_q(i,j) = new_val_f2_12(9);  
                    elseif G_f2_12(i,j) >= edges(10) && G_f2_12(i,j) < edges(11)  
                        G_f2_12_q(i,j) = new_val_f2_12(10);  
                    elseif G_f2_12(i,j) >= edges(11) && G_f2_12(i,j) < edges(12)  
                        G_f2_12_q(i,j) = new_val_f2_12(11);  
                    elseif G_f2_12(i,j) >= edges(12) && G_f2_12(i,j) < edges(13)  
                        G_f2_12_q(i,j) = new_val_f2_12(12);  
                    elseif G_f2_12(i,j) >= edges(13) && G_f2_12(i,j) < edges(14)   
                         G_f2_12_q(i,j) = new_val_f2_12(13);   
                    elseif G_f2_12(i,j) >= edges(14) && G_f2_12(i,j) < edges(15)  
                        G_f2_12_q(i,j) = new_val_f2_12(14);  
                    elseif G_f2_12(i,j) >= edges(15) && G_f2_12(i,j) < edges(16)  
                        G_f2_12_q(i,j) = new_val_f2_12(15);  
                    elseif G_f2_12(i,j) >= edges(16) && G_f2_12(i,j) <= edges(17)   
                         G_f2_12_q(i,j) = new_val_f2_12(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_13,1) 
                for j = 1:size(G_f2_13,2) 
                    if G_f2_13(i,j) < edges(1)  
                        G_f2_13_q(i,j) = new_val_f2_13(1);  
                    elseif G_f2_13(i,j) >= edges(1) && G_f2_13(i,j) < edges(2)  
                        G_f2_13_q(i,j) = new_val_f2_13(1);  
                    elseif G_f2_13(i,j) >= edges(2) && G_f2_13(i,j) < edges(3)  
                        G_f2_13_q(i,j) = new_val_f2_13(2);    
                    elseif G_f2_13(i,j) >= edges(3) && G_f2_13(i,j) < edges(4)  
                        G_f2_13_q(i,j) = new_val_f2_13(3);  
                    elseif G_f2_13(i,j) >= edges(4) && G_f2_13(i,j) < edges(5)  
                        G_f2_13_q(i,j) = new_val_f2_13(4);   
                    elseif G_f2_13(i,j) >= edges(5) && G_f2_13(i,j) < edges(6)  
                        G_f2_13_q(i,j) = new_val_f2_13(5);     
                    elseif G_f2_13(i,j) >= edges(6) && G_f2_13(i,j) < edges(7)  
                        G_f2_13_q(i,j) = new_val_f2_13(6);  
                    elseif G_f2_13(i,j) >= edges(7) && G_f2_13(i,j) < edges(8)  
                        G_f2_13_q(i,j) = new_val_f2_13(7);   
                    elseif G_f2_13(i,j) >= edges(8) && G_f2_13(i,j) < edges(9)  
                        G_f2_13_q(i,j) = new_val_f2_13(8);     
                    elseif G_f2_13(i,j) >= edges(9) && G_f2_13(i,j) < edges(10)  
                        G_f2_13_q(i,j) = new_val_f2_13(9);  
                    elseif G_f2_13(i,j) >= edges(10) && G_f2_13(i,j) < edges(11)  
                        G_f2_13_q(i,j) = new_val_f2_13(10);  
                    elseif G_f2_13(i,j) >= edges(11) && G_f2_13(i,j) < edges(12)  
                        G_f2_13_q(i,j) = new_val_f2_13(11);  
                    elseif G_f2_13(i,j) >= edges(12) && G_f2_13(i,j) < edges(13)  
                        G_f2_13_q(i,j) = new_val_f2_13(12);  
                    elseif G_f2_13(i,j) >= edges(13) && G_f2_13(i,j) < edges(14)   
                         G_f2_13_q(i,j) = new_val_f2_13(13);   
                    elseif G_f2_13(i,j) >= edges(14) && G_f2_13(i,j) < edges(15)  
                        G_f2_13_q(i,j) = new_val_f2_13(14);  
                    elseif G_f2_13(i,j) >= edges(15) && G_f2_13(i,j) < edges(16)  
                        G_f2_13_q(i,j) = new_val_f2_13(15);  
                    elseif G_f2_13(i,j) >= edges(16) && G_f2_13(i,j) <= edges(17)   
                         G_f2_13_q(i,j) = new_val_f2_13(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_14,1) 
                for j = 1:size(G_f2_14,2) 
                    if G_f2_14(i,j) < edges(1)  
                        G_f2_14_q(i,j) = new_val_f2_14(1);  
                    elseif G_f2_14(i,j) >= edges(1) && G_f2_14(i,j) < edges(2)  
                        G_f2_14_q(i,j) = new_val_f2_14(1);  
                    elseif G_f2_14(i,j) >= edges(2) && G_f2_14(i,j) < edges(3)  
                        G_f2_14_q(i,j) = new_val_f2_14(2);    
                    elseif G_f2_14(i,j) >= edges(3) && G_f2_14(i,j) < edges(4)  
                        G_f2_14_q(i,j) = new_val_f2_14(3);  
                    elseif G_f2_14(i,j) >= edges(4) && G_f2_14(i,j) < edges(5)  
                        G_f2_14_q(i,j) = new_val_f2_14(4);   
                    elseif G_f2_14(i,j) >= edges(5) && G_f2_14(i,j) < edges(6)  
                        G_f2_14_q(i,j) = new_val_f2_14(5);     
                    elseif G_f2_14(i,j) >= edges(6) && G_f2_14(i,j) < edges(7)  
                        G_f2_14_q(i,j) = new_val_f2_14(6);  
                    elseif G_f2_14(i,j) >= edges(7) && G_f2_14(i,j) < edges(8)  
                        G_f2_14_q(i,j) = new_val_f2_14(7);   
                    elseif G_f2_14(i,j) >= edges(8) && G_f2_14(i,j) < edges(9)  
                        G_f2_14_q(i,j) = new_val_f2_14(8);     
                    elseif G_f2_14(i,j) >= edges(9) && G_f2_14(i,j) < edges(10)  
                        G_f2_14_q(i,j) = new_val_f2_14(9);  
                    elseif G_f2_14(i,j) >= edges(10) && G_f2_14(i,j) < edges(11)  
                        G_f2_14_q(i,j) = new_val_f2_14(10);  
                    elseif G_f2_14(i,j) >= edges(11) && G_f2_14(i,j) < edges(12)  
                        G_f2_14_q(i,j) = new_val_f2_14(11);  
                    elseif G_f2_14(i,j) >= edges(12) && G_f2_14(i,j) < edges(13)  
                        G_f2_14_q(i,j) = new_val_f2_14(12);  
                    elseif G_f2_14(i,j) >= edges(13) && G_f2_14(i,j) < edges(14)   
                         G_f2_14_q(i,j) = new_val_f2_14(13);   
                    elseif G_f2_14(i,j) >= edges(14) && G_f2_14(i,j) < edges(15)  
                        G_f2_14_q(i,j) = new_val_f2_14(14);  
                    elseif G_f2_14(i,j) >= edges(15) && G_f2_14(i,j) < edges(16)  
                        G_f2_14_q(i,j) = new_val_f2_14(15);  
                    elseif G_f2_14(i,j) >= edges(16) && G_f2_14(i,j) <= edges(17)   
                         G_f2_14_q(i,j) = new_val_f2_14(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_15,1) 
                for j = 1:size(G_f2_15,2) 
                    if G_f2_15(i,j) < edges(1)  
                        G_f2_15_q(i,j) = new_val_f2_15(1);  
                    elseif G_f2_15(i,j) >= edges(1) && G_f2_15(i,j) < edges(2)  
                        G_f2_15_q(i,j) = new_val_f2_15(1);  
                    elseif G_f2_15(i,j) >= edges(2) && G_f2_15(i,j) < edges(3)  
                        G_f2_15_q(i,j) = new_val_f2_15(2);    
                    elseif G_f2_15(i,j) >= edges(3) && G_f2_15(i,j) < edges(4)  
                        G_f2_15_q(i,j) = new_val_f2_15(3);  
                    elseif G_f2_15(i,j) >= edges(4) && G_f2_15(i,j) < edges(5)  
                        G_f2_15_q(i,j) = new_val_f2_15(4);   
                    elseif G_f2_15(i,j) >= edges(5) && G_f2_15(i,j) < edges(6)  
                        G_f2_15_q(i,j) = new_val_f2_15(5);     
                    elseif G_f2_15(i,j) >= edges(6) && G_f2_15(i,j) < edges(7)  
                        G_f2_15_q(i,j) = new_val_f2_15(6);  
                    elseif G_f2_15(i,j) >= edges(7) && G_f2_15(i,j) < edges(8)  
                        G_f2_15_q(i,j) = new_val_f2_15(7);   
                    elseif G_f2_15(i,j) >= edges(8) && G_f2_15(i,j) < edges(9)  
                        G_f2_15_q(i,j) = new_val_f2_15(8);     
                    elseif G_f2_15(i,j) >= edges(9) && G_f2_15(i,j) < edges(10)  
                        G_f2_15_q(i,j) = new_val_f2_15(9);  
                    elseif G_f2_15(i,j) >= edges(10) && G_f2_15(i,j) < edges(11)  
                        G_f2_15_q(i,j) = new_val_f2_15(10);  
                    elseif G_f2_15(i,j) >= edges(11) && G_f2_15(i,j) < edges(12)  
                        G_f2_15_q(i,j) = new_val_f2_15(11);  
                    elseif G_f2_15(i,j) >= edges(12) && G_f2_15(i,j) < edges(13)  
                        G_f2_15_q(i,j) = new_val_f2_15(12);  
                    elseif G_f2_15(i,j) >= edges(13) && G_f2_15(i,j) < edges(14)   
                         G_f2_15_q(i,j) = new_val_f2_15(13);   
                    elseif G_f2_15(i,j) >= edges(14) && G_f2_15(i,j) < edges(15)  
                        G_f2_15_q(i,j) = new_val_f2_15(14);  
                    elseif G_f2_15(i,j) >= edges(15) && G_f2_15(i,j) < edges(16)  
                        G_f2_15_q(i,j) = new_val_f2_15(15);  
                    elseif G_f2_15(i,j) >= edges(16) && G_f2_15(i,j) <= edges(17)   
                         G_f2_15_q(i,j) = new_val_f2_15(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_16,1) 
                for j = 1:size(G_f2_16,2) 
                    if G_f2_16(i,j) < edges(1)  
                        G_f2_16_q(i,j) = new_val_f2_16(1);  
                    elseif G_f2_16(i,j) >= edges(1) && G_f2_16(i,j) < edges(2)  
                        G_f2_16_q(i,j) = new_val_f2_16(1);  
                    elseif G_f2_16(i,j) >= edges(2) && G_f2_16(i,j) < edges(3)  
                        G_f2_16_q(i,j) = new_val_f2_16(2);    
                    elseif G_f2_16(i,j) >= edges(3) && G_f2_16(i,j) < edges(4)  
                        G_f2_16_q(i,j) = new_val_f2_16(3);  
                    elseif G_f2_16(i,j) >= edges(4) && G_f2_16(i,j) < edges(5)  
                        G_f2_16_q(i,j) = new_val_f2_16(4);   
                    elseif G_f2_16(i,j) >= edges(5) && G_f2_16(i,j) < edges(6)  
                        G_f2_16_q(i,j) = new_val_f2_16(5);     
                    elseif G_f2_16(i,j) >= edges(6) && G_f2_16(i,j) < edges(7)  
                        G_f2_16_q(i,j) = new_val_f2_16(6);  
                    elseif G_f2_16(i,j) >= edges(7) && G_f2_16(i,j) < edges(8)  
                        G_f2_16_q(i,j) = new_val_f2_16(7);   
                    elseif G_f2_16(i,j) >= edges(8) && G_f2_16(i,j) < edges(9)  
                        G_f2_16_q(i,j) = new_val_f2_16(8);     
                    elseif G_f2_16(i,j) >= edges(9) && G_f2_16(i,j) < edges(10)  
                        G_f2_16_q(i,j) = new_val_f2_16(9);  
                    elseif G_f2_16(i,j) >= edges(10) && G_f2_16(i,j) < edges(11)  
                        G_f2_16_q(i,j) = new_val_f2_16(10);  
                    elseif G_f2_16(i,j) >= edges(11) && G_f2_16(i,j) < edges(12)  
                        G_f2_16_q(i,j) = new_val_f2_16(11);  
                    elseif G_f2_16(i,j) >= edges(12) && G_f2_16(i,j) < edges(13)  
                        G_f2_16_q(i,j) = new_val_f2_16(12);  
                    elseif G_f2_16(i,j) >= edges(13) && G_f2_16(i,j) < edges(14)   
                         G_f2_16_q(i,j) = new_val_f2_16(13);   
                    elseif G_f2_16(i,j) >= edges(14) && G_f2_16(i,j) < edges(15)  
                        G_f2_16_q(i,j) = new_val_f2_16(14);  
                    elseif G_f2_16(i,j) >= edges(15) && G_f2_16(i,j) < edges(16)  
                        G_f2_16_q(i,j) = new_val_f2_16(15);  
                    elseif G_f2_16(i,j) >= edges(16) && G_f2_16(i,j) <= edges(17)   
                         G_f2_16_q(i,j) = new_val_f2_16(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_17,1) 
                for j = 1:size(G_f2_17,2) 
                    if G_f2_17(i,j) < edges(1)  
                        G_f2_17_q(i,j) = new_val_f2_17(1);  
                    elseif G_f2_17(i,j) >= edges(1) && G_f2_17(i,j) < edges(2)  
                        G_f2_17_q(i,j) = new_val_f2_17(1);  
                    elseif G_f2_17(i,j) >= edges(2) && G_f2_17(i,j) < edges(3)  
                        G_f2_17_q(i,j) = new_val_f2_17(2);    
                    elseif G_f2_17(i,j) >= edges(3) && G_f2_17(i,j) < edges(4)  
                        G_f2_17_q(i,j) = new_val_f2_17(3);  
                    elseif G_f2_17(i,j) >= edges(4) && G_f2_17(i,j) < edges(5)  
                        G_f2_17_q(i,j) = new_val_f2_17(4);   
                    elseif G_f2_17(i,j) >= edges(5) && G_f2_17(i,j) < edges(6)  
                        G_f2_17_q(i,j) = new_val_f2_17(5);     
                    elseif G_f2_17(i,j) >= edges(6) && G_f2_17(i,j) < edges(7)  
                        G_f2_17_q(i,j) = new_val_f2_17(6);  
                    elseif G_f2_17(i,j) >= edges(7) && G_f2_17(i,j) < edges(8)  
                        G_f2_17_q(i,j) = new_val_f2_17(7);   
                    elseif G_f2_17(i,j) >= edges(8) && G_f2_17(i,j) < edges(9)  
                        G_f2_17_q(i,j) = new_val_f2_17(8);     
                    elseif G_f2_17(i,j) >= edges(9) && G_f2_17(i,j) < edges(10)  
                        G_f2_17_q(i,j) = new_val_f2_17(9);  
                    elseif G_f2_17(i,j) >= edges(10) && G_f2_17(i,j) < edges(11)  
                        G_f2_17_q(i,j) = new_val_f2_17(10);  
                    elseif G_f2_17(i,j) >= edges(11) && G_f2_17(i,j) < edges(12)  
                        G_f2_17_q(i,j) = new_val_f2_17(11);  
                    elseif G_f2_17(i,j) >= edges(12) && G_f2_17(i,j) < edges(13)  
                        G_f2_17_q(i,j) = new_val_f2_17(12);  
                    elseif G_f2_17(i,j) >= edges(13) && G_f2_17(i,j) < edges(14)   
                         G_f2_17_q(i,j) = new_val_f2_17(13);   
                    elseif G_f2_17(i,j) >= edges(14) && G_f2_17(i,j) < edges(15)  
                        G_f2_17_q(i,j) = new_val_f2_17(14);  
                    elseif G_f2_17(i,j) >= edges(15) && G_f2_17(i,j) < edges(16)  
                        G_f2_17_q(i,j) = new_val_f2_17(15);  
                    elseif G_f2_17(i,j) >= edges(16) && G_f2_17(i,j) <= edges(17)   
                         G_f2_17_q(i,j) = new_val_f2_17(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_18,1) 
                for j = 1:size(G_f2_18,2) 
                    if G_f2_18(i,j) < edges(1)  
                        G_f2_18_q(i,j) = new_val_f2_18(1);  
                    elseif G_f2_18(i,j) >= edges(1) && G_f2_18(i,j) < edges(2)  
                        G_f2_18_q(i,j) = new_val_f2_18(1);  
                    elseif G_f2_18(i,j) >= edges(2) && G_f2_18(i,j) < edges(3)  
                        G_f2_18_q(i,j) = new_val_f2_18(2);    
                    elseif G_f2_18(i,j) >= edges(3) && G_f2_18(i,j) < edges(4)  
                        G_f2_18_q(i,j) = new_val_f2_18(3);  
                    elseif G_f2_18(i,j) >= edges(4) && G_f2_18(i,j) < edges(5)  
                        G_f2_18_q(i,j) = new_val_f2_18(4);   
                    elseif G_f2_18(i,j) >= edges(5) && G_f2_18(i,j) < edges(6)  
                        G_f2_18_q(i,j) = new_val_f2_18(5);     
                    elseif G_f2_18(i,j) >= edges(6) && G_f2_18(i,j) < edges(7)  
                        G_f2_18_q(i,j) = new_val_f2_18(6);  
                    elseif G_f2_18(i,j) >= edges(7) && G_f2_18(i,j) < edges(8)  
                        G_f2_18_q(i,j) = new_val_f2_18(7);   
                    elseif G_f2_18(i,j) >= edges(8) && G_f2_18(i,j) < edges(9)  
                        G_f2_18_q(i,j) = new_val_f2_18(8);     
                    elseif G_f2_18(i,j) >= edges(9) && G_f2_18(i,j) < edges(10)  
                        G_f2_18_q(i,j) = new_val_f2_18(9);  
                    elseif G_f2_18(i,j) >= edges(10) && G_f2_18(i,j) < edges(11)  
                        G_f2_18_q(i,j) = new_val_f2_18(10);  
                    elseif G_f2_18(i,j) >= edges(11) && G_f2_18(i,j) < edges(12)  
                        G_f2_18_q(i,j) = new_val_f2_18(11);  
                    elseif G_f2_18(i,j) >= edges(12) && G_f2_18(i,j) < edges(13)  
                        G_f2_18_q(i,j) = new_val_f2_18(12);  
                    elseif G_f2_18(i,j) >= edges(13) && G_f2_18(i,j) < edges(14)   
                         G_f2_18_q(i,j) = new_val_f2_18(13);   
                    elseif G_f2_18(i,j) >= edges(14) && G_f2_18(i,j) < edges(15)  
                        G_f2_18_q(i,j) = new_val_f2_18(14);  
                    elseif G_f2_18(i,j) >= edges(15) && G_f2_18(i,j) < edges(16)  
                        G_f2_18_q(i,j) = new_val_f2_18(15);  
                    elseif G_f2_18(i,j) >= edges(16) && G_f2_18(i,j) <= edges(17)   
                         G_f2_18_q(i,j) = new_val_f2_18(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_19,1) 
                for j = 1:size(G_f2_19,2) 
                    if G_f2_19(i,j) < edges(1)  
                        G_f2_19_q(i,j) = new_val_f2_19(1);  
                    elseif G_f2_19(i,j) >= edges(1) && G_f2_19(i,j) < edges(2)  
                        G_f2_19_q(i,j) = new_val_f2_19(1);  
                    elseif G_f2_19(i,j) >= edges(2) && G_f2_19(i,j) < edges(3)  
                        G_f2_19_q(i,j) = new_val_f2_19(2);    
                    elseif G_f2_19(i,j) >= edges(3) && G_f2_19(i,j) < edges(4)  
                        G_f2_19_q(i,j) = new_val_f2_19(3);  
                    elseif G_f2_19(i,j) >= edges(4) && G_f2_19(i,j) < edges(5)  
                        G_f2_19_q(i,j) = new_val_f2_19(4);   
                    elseif G_f2_19(i,j) >= edges(5) && G_f2_19(i,j) < edges(6)  
                        G_f2_19_q(i,j) = new_val_f2_19(5);     
                    elseif G_f2_19(i,j) >= edges(6) && G_f2_19(i,j) < edges(7)  
                        G_f2_19_q(i,j) = new_val_f2_19(6);  
                    elseif G_f2_19(i,j) >= edges(7) && G_f2_19(i,j) < edges(8)  
                        G_f2_19_q(i,j) = new_val_f2_19(7);   
                    elseif G_f2_19(i,j) >= edges(8) && G_f2_19(i,j) < edges(9)  
                        G_f2_19_q(i,j) = new_val_f2_19(8);     
                    elseif G_f2_19(i,j) >= edges(9) && G_f2_19(i,j) < edges(10)  
                        G_f2_19_q(i,j) = new_val_f2_19(9);  
                    elseif G_f2_19(i,j) >= edges(10) && G_f2_19(i,j) < edges(11)  
                        G_f2_19_q(i,j) = new_val_f2_19(10);  
                    elseif G_f2_19(i,j) >= edges(11) && G_f2_19(i,j) < edges(12)  
                        G_f2_19_q(i,j) = new_val_f2_19(11);  
                    elseif G_f2_19(i,j) >= edges(12) && G_f2_19(i,j) < edges(13)  
                        G_f2_19_q(i,j) = new_val_f2_19(12);  
                    elseif G_f2_19(i,j) >= edges(13) && G_f2_19(i,j) < edges(14)   
                         G_f2_19_q(i,j) = new_val_f2_19(13);   
                    elseif G_f2_19(i,j) >= edges(14) && G_f2_19(i,j) < edges(15)  
                        G_f2_19_q(i,j) = new_val_f2_19(14);  
                    elseif G_f2_19(i,j) >= edges(15) && G_f2_19(i,j) < edges(16)  
                        G_f2_19_q(i,j) = new_val_f2_19(15);  
                    elseif G_f2_19(i,j) >= edges(16) && G_f2_19(i,j) <= edges(17)   
                         G_f2_19_q(i,j) = new_val_f2_19(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_20,1) 
                for j = 1:size(G_f2_20,2) 
                    if G_f2_20(i,j) < edges(1)  
                        G_f2_20_q(i,j) = new_val_f2_20(1);  
                    elseif G_f2_20(i,j) >= edges(1) && G_f2_20(i,j) < edges(2)  
                        G_f2_20_q(i,j) = new_val_f2_20(1);  
                    elseif G_f2_20(i,j) >= edges(2) && G_f2_20(i,j) < edges(3)  
                        G_f2_20_q(i,j) = new_val_f2_20(2);    
                    elseif G_f2_20(i,j) >= edges(3) && G_f2_20(i,j) < edges(4)  
                        G_f2_20_q(i,j) = new_val_f2_20(3);  
                    elseif G_f2_20(i,j) >= edges(4) && G_f2_20(i,j) < edges(5)  
                        G_f2_20_q(i,j) = new_val_f2_20(4);   
                    elseif G_f2_20(i,j) >= edges(5) && G_f2_20(i,j) < edges(6)  
                        G_f2_20_q(i,j) = new_val_f2_20(5);     
                    elseif G_f2_20(i,j) >= edges(6) && G_f2_20(i,j) < edges(7)  
                        G_f2_20_q(i,j) = new_val_f2_20(6);  
                    elseif G_f2_20(i,j) >= edges(7) && G_f2_20(i,j) < edges(8)  
                        G_f2_20_q(i,j) = new_val_f2_20(7);   
                    elseif G_f2_20(i,j) >= edges(8) && G_f2_20(i,j) < edges(9)  
                        G_f2_20_q(i,j) = new_val_f2_20(8);     
                    elseif G_f2_20(i,j) >= edges(9) && G_f2_20(i,j) < edges(10)  
                        G_f2_20_q(i,j) = new_val_f2_20(9);  
                    elseif G_f2_20(i,j) >= edges(10) && G_f2_20(i,j) < edges(11)  
                        G_f2_20_q(i,j) = new_val_f2_20(10);  
                    elseif G_f2_20(i,j) >= edges(11) && G_f2_20(i,j) < edges(12)  
                        G_f2_20_q(i,j) = new_val_f2_20(11);  
                    elseif G_f2_20(i,j) >= edges(12) && G_f2_20(i,j) < edges(13)  
                        G_f2_20_q(i,j) = new_val_f2_20(12);  
                    elseif G_f2_20(i,j) >= edges(13) && G_f2_20(i,j) < edges(14)   
                         G_f2_20_q(i,j) = new_val_f2_20(13);   
                    elseif G_f2_20(i,j) >= edges(14) && G_f2_20(i,j) < edges(15)  
                        G_f2_20_q(i,j) = new_val_f2_20(14);  
                    elseif G_f2_20(i,j) >= edges(15) && G_f2_20(i,j) < edges(16)  
                        G_f2_20_q(i,j) = new_val_f2_20(15);  
                    elseif G_f2_20(i,j) >= edges(16) && G_f2_20(i,j) <= edges(17)   
                         G_f2_20_q(i,j) = new_val_f2_20(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_21,1) 
                for j = 1:size(G_f2_21,2) 
                    if G_f2_21(i,j) < edges(1)  
                        G_f2_21_q(i,j) = new_val_f2_21(1);  
                    elseif G_f2_21(i,j) >= edges(1) && G_f2_21(i,j) < edges(2)  
                        G_f2_21_q(i,j) = new_val_f2_21(1);  
                    elseif G_f2_21(i,j) >= edges(2) && G_f2_21(i,j) < edges(3)  
                        G_f2_21_q(i,j) = new_val_f2_21(2);    
                    elseif G_f2_21(i,j) >= edges(3) && G_f2_21(i,j) < edges(4)  
                        G_f2_21_q(i,j) = new_val_f2_21(3);  
                    elseif G_f2_21(i,j) >= edges(4) && G_f2_21(i,j) < edges(5)  
                        G_f2_21_q(i,j) = new_val_f2_21(4);   
                    elseif G_f2_21(i,j) >= edges(5) && G_f2_21(i,j) < edges(6)  
                        G_f2_21_q(i,j) = new_val_f2_21(5);     
                    elseif G_f2_21(i,j) >= edges(6) && G_f2_21(i,j) < edges(7)  
                        G_f2_21_q(i,j) = new_val_f2_21(6);  
                    elseif G_f2_21(i,j) >= edges(7) && G_f2_21(i,j) < edges(8)  
                        G_f2_21_q(i,j) = new_val_f2_21(7);   
                    elseif G_f2_21(i,j) >= edges(8) && G_f2_21(i,j) < edges(9)  
                        G_f2_21_q(i,j) = new_val_f2_21(8);     
                    elseif G_f2_21(i,j) >= edges(9) && G_f2_21(i,j) < edges(10)  
                        G_f2_21_q(i,j) = new_val_f2_21(9);  
                    elseif G_f2_21(i,j) >= edges(10) && G_f2_21(i,j) < edges(11)  
                        G_f2_21_q(i,j) = new_val_f2_21(10);  
                    elseif G_f2_21(i,j) >= edges(11) && G_f2_21(i,j) < edges(12)  
                        G_f2_21_q(i,j) = new_val_f2_21(11);  
                    elseif G_f2_21(i,j) >= edges(12) && G_f2_21(i,j) < edges(13)  
                        G_f2_21_q(i,j) = new_val_f2_21(12);  
                    elseif G_f2_21(i,j) >= edges(13) && G_f2_21(i,j) < edges(14)   
                         G_f2_21_q(i,j) = new_val_f2_21(13);   
                    elseif G_f2_21(i,j) >= edges(14) && G_f2_21(i,j) < edges(15)  
                        G_f2_21_q(i,j) = new_val_f2_21(14);  
                    elseif G_f2_21(i,j) >= edges(15) && G_f2_21(i,j) < edges(16)  
                        G_f2_21_q(i,j) = new_val_f2_21(15);  
                    elseif G_f2_21(i,j) >= edges(16) && G_f2_21(i,j) <= edges(17)   
                         G_f2_21_q(i,j) = new_val_f2_21(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_22,1) 
                for j = 1:size(G_f2_22,2) 
                    if G_f2_22(i,j) < edges(1)  
                        G_f2_22_q(i,j) = new_val_f2_22(1);  
                    elseif G_f2_22(i,j) >= edges(1) && G_f2_22(i,j) < edges(2)  
                        G_f2_22_q(i,j) = new_val_f2_22(1);  
                    elseif G_f2_22(i,j) >= edges(2) && G_f2_22(i,j) < edges(3)  
                        G_f2_22_q(i,j) = new_val_f2_22(2);    
                    elseif G_f2_22(i,j) >= edges(3) && G_f2_22(i,j) < edges(4)  
                        G_f2_22_q(i,j) = new_val_f2_22(3);  
                    elseif G_f2_22(i,j) >= edges(4) && G_f2_22(i,j) < edges(5)  
                        G_f2_22_q(i,j) = new_val_f2_22(4);   
                    elseif G_f2_22(i,j) >= edges(5) && G_f2_22(i,j) < edges(6)  
                        G_f2_22_q(i,j) = new_val_f2_22(5);     
                    elseif G_f2_22(i,j) >= edges(6) && G_f2_22(i,j) < edges(7)  
                        G_f2_22_q(i,j) = new_val_f2_22(6);  
                    elseif G_f2_22(i,j) >= edges(7) && G_f2_22(i,j) < edges(8)  
                        G_f2_22_q(i,j) = new_val_f2_22(7);   
                    elseif G_f2_22(i,j) >= edges(8) && G_f2_22(i,j) < edges(9)  
                        G_f2_22_q(i,j) = new_val_f2_22(8);     
                    elseif G_f2_22(i,j) >= edges(9) && G_f2_22(i,j) < edges(10)  
                        G_f2_22_q(i,j) = new_val_f2_22(9);  
                    elseif G_f2_22(i,j) >= edges(10) && G_f2_22(i,j) < edges(11)  
                        G_f2_22_q(i,j) = new_val_f2_22(10);  
                    elseif G_f2_22(i,j) >= edges(11) && G_f2_22(i,j) < edges(12)  
                        G_f2_22_q(i,j) = new_val_f2_22(11);  
                    elseif G_f2_22(i,j) >= edges(12) && G_f2_22(i,j) < edges(13)  
                        G_f2_22_q(i,j) = new_val_f2_22(12);  
                    elseif G_f2_22(i,j) >= edges(13) && G_f2_22(i,j) < edges(14)   
                         G_f2_22_q(i,j) = new_val_f2_22(13);   
                    elseif G_f2_22(i,j) >= edges(14) && G_f2_22(i,j) < edges(15)  
                        G_f2_22_q(i,j) = new_val_f2_22(14);  
                    elseif G_f2_22(i,j) >= edges(15) && G_f2_22(i,j) < edges(16)  
                        G_f2_22_q(i,j) = new_val_f2_22(15);  
                    elseif G_f2_22(i,j) >= edges(16) && G_f2_22(i,j) <= edges(17)   
                         G_f2_22_q(i,j) = new_val_f2_22(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_23,1) 
                for j = 1:size(G_f2_23,2) 
                    if G_f2_23(i,j) < edges(1)  
                        G_f2_23_q(i,j) = new_val_f2_23(1);  
                    elseif G_f2_23(i,j) >= edges(1) && G_f2_23(i,j) < edges(2)  
                        G_f2_23_q(i,j) = new_val_f2_23(1);  
                    elseif G_f2_23(i,j) >= edges(2) && G_f2_23(i,j) < edges(3)  
                        G_f2_23_q(i,j) = new_val_f2_23(2);    
                    elseif G_f2_23(i,j) >= edges(3) && G_f2_23(i,j) < edges(4)  
                        G_f2_23_q(i,j) = new_val_f2_23(3);  
                    elseif G_f2_23(i,j) >= edges(4) && G_f2_23(i,j) < edges(5)  
                        G_f2_23_q(i,j) = new_val_f2_23(4);   
                    elseif G_f2_23(i,j) >= edges(5) && G_f2_23(i,j) < edges(6)  
                        G_f2_23_q(i,j) = new_val_f2_23(5);     
                    elseif G_f2_23(i,j) >= edges(6) && G_f2_23(i,j) < edges(7)  
                        G_f2_23_q(i,j) = new_val_f2_23(6);  
                    elseif G_f2_23(i,j) >= edges(7) && G_f2_23(i,j) < edges(8)  
                        G_f2_23_q(i,j) = new_val_f2_23(7);   
                    elseif G_f2_23(i,j) >= edges(8) && G_f2_23(i,j) < edges(9)  
                        G_f2_23_q(i,j) = new_val_f2_23(8);     
                    elseif G_f2_23(i,j) >= edges(9) && G_f2_23(i,j) < edges(10)  
                        G_f2_23_q(i,j) = new_val_f2_23(9);  
                    elseif G_f2_23(i,j) >= edges(10) && G_f2_23(i,j) < edges(11)  
                        G_f2_23_q(i,j) = new_val_f2_23(10);  
                    elseif G_f2_23(i,j) >= edges(11) && G_f2_23(i,j) < edges(12)  
                        G_f2_23_q(i,j) = new_val_f2_23(11);  
                    elseif G_f2_23(i,j) >= edges(12) && G_f2_23(i,j) < edges(13)  
                        G_f2_23_q(i,j) = new_val_f2_23(12);  
                    elseif G_f2_23(i,j) >= edges(13) && G_f2_23(i,j) < edges(14)   
                         G_f2_23_q(i,j) = new_val_f2_23(13);   
                    elseif G_f2_23(i,j) >= edges(14) && G_f2_23(i,j) < edges(15)  
                        G_f2_23_q(i,j) = new_val_f2_23(14);  
                    elseif G_f2_23(i,j) >= edges(15) && G_f2_23(i,j) < edges(16)  
                        G_f2_23_q(i,j) = new_val_f2_23(15);  
                    elseif G_f2_23(i,j) >= edges(16) && G_f2_23(i,j) <= edges(17)   
                         G_f2_23_q(i,j) = new_val_f2_23(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_24,1) 
                for j = 1:size(G_f2_24,2) 
                    if G_f2_24(i,j) < edges(1)  
                        G_f2_24_q(i,j) = new_val_f2_24(1);  
                    elseif G_f2_24(i,j) >= edges(1) && G_f2_24(i,j) < edges(2)  
                        G_f2_24_q(i,j) = new_val_f2_24(1);  
                    elseif G_f2_24(i,j) >= edges(2) && G_f2_24(i,j) < edges(3)  
                        G_f2_24_q(i,j) = new_val_f2_24(2);    
                    elseif G_f2_24(i,j) >= edges(3) && G_f2_24(i,j) < edges(4)  
                        G_f2_24_q(i,j) = new_val_f2_24(3);  
                    elseif G_f2_24(i,j) >= edges(4) && G_f2_24(i,j) < edges(5)  
                        G_f2_24_q(i,j) = new_val_f2_24(4);   
                    elseif G_f2_24(i,j) >= edges(5) && G_f2_24(i,j) < edges(6)  
                        G_f2_24_q(i,j) = new_val_f2_24(5);     
                    elseif G_f2_24(i,j) >= edges(6) && G_f2_24(i,j) < edges(7)  
                        G_f2_24_q(i,j) = new_val_f2_24(6);  
                    elseif G_f2_24(i,j) >= edges(7) && G_f2_24(i,j) < edges(8)  
                        G_f2_24_q(i,j) = new_val_f2_24(7);   
                    elseif G_f2_24(i,j) >= edges(8) && G_f2_24(i,j) < edges(9)  
                        G_f2_24_q(i,j) = new_val_f2_24(8);     
                    elseif G_f2_24(i,j) >= edges(9) && G_f2_24(i,j) < edges(10)  
                        G_f2_24_q(i,j) = new_val_f2_24(9);  
                    elseif G_f2_24(i,j) >= edges(10) && G_f2_24(i,j) < edges(11)  
                        G_f2_24_q(i,j) = new_val_f2_24(10);  
                    elseif G_f2_24(i,j) >= edges(11) && G_f2_24(i,j) < edges(12)  
                        G_f2_24_q(i,j) = new_val_f2_24(11);  
                    elseif G_f2_24(i,j) >= edges(12) && G_f2_24(i,j) <= edges(13)  
                        G_f2_24_q(i,j) = new_val_f2_24(12);  
                    elseif G_f2_24(i,j) >= edges(13) && G_f2_24(i,j) < edges(14)   
                         G_f2_24_q(i,j) = new_val_f2_24(13);   
                    elseif G_f2_24(i,j) >= edges(14) && G_f2_24(i,j) < edges(15)  
                        G_f2_24_q(i,j) = new_val_f2_24(14);  
                    elseif G_f2_24(i,j) >= edges(15) && G_f2_24(i,j) < edges(16)  
                        G_f2_24_q(i,j) = new_val_f2_24(15);  
                    elseif G_f2_24(i,j) >= edges(16) && G_f2_24(i,j) <= edges(17)   
                         G_f2_24_q(i,j) = new_val_f2_24(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_25,1) 
                for j = 1:size(G_f2_25,2) 
                    if G_f2_25(i,j) < edges(1)  
                        G_f2_25_q(i,j) = new_val_f2_25(1);  
                    elseif G_f2_25(i,j) >= edges(1) && G_f2_25(i,j) < edges(2)  
                        G_f2_25_q(i,j) = new_val_f2_25(1);  
                    elseif G_f2_25(i,j) >= edges(2) && G_f2_25(i,j) < edges(3)  
                        G_f2_25_q(i,j) = new_val_f2_25(2);    
                    elseif G_f2_25(i,j) >= edges(3) && G_f2_25(i,j) < edges(4)  
                        G_f2_25_q(i,j) = new_val_f2_25(3);  
                    elseif G_f2_25(i,j) >= edges(4) && G_f2_25(i,j) < edges(5)  
                        G_f2_25_q(i,j) = new_val_f2_25(4);   
                    elseif G_f2_25(i,j) >= edges(5) && G_f2_25(i,j) < edges(6)  
                        G_f2_25_q(i,j) = new_val_f2_25(5);     
                    elseif G_f2_25(i,j) >= edges(6) && G_f2_25(i,j) < edges(7)  
                        G_f2_25_q(i,j) = new_val_f2_25(6);  
                    elseif G_f2_25(i,j) >= edges(7) && G_f2_25(i,j) < edges(8)  
                        G_f2_25_q(i,j) = new_val_f2_25(7);   
                    elseif G_f2_25(i,j) >= edges(8) && G_f2_25(i,j) < edges(9)  
                        G_f2_25_q(i,j) = new_val_f2_25(8);     
                    elseif G_f2_25(i,j) >= edges(9) && G_f2_25(i,j) < edges(10)  
                        G_f2_25_q(i,j) = new_val_f2_25(9);  
                    elseif G_f2_25(i,j) >= edges(10) && G_f2_25(i,j) < edges(11)  
                        G_f2_25_q(i,j) = new_val_f2_25(10);  
                    elseif G_f2_25(i,j) >= edges(11) && G_f2_25(i,j) < edges(12)  
                        G_f2_25_q(i,j) = new_val_f2_25(11);  
                    elseif G_f2_25(i,j) >= edges(12) && G_f2_25(i,j) < edges(13)  
                        G_f2_25_q(i,j) = new_val_f2_25(12);  
                    elseif G_f2_25(i,j) >= edges(13) && G_f2_25(i,j) < edges(14)   
                         G_f2_25_q(i,j) = new_val_f2_25(13);   
                    elseif G_f2_25(i,j) >= edges(14) && G_f2_25(i,j) < edges(15)  
                        G_f2_25_q(i,j) = new_val_f2_25(14);  
                    elseif G_f2_25(i,j) >= edges(15) && G_f2_25(i,j) < edges(16)  
                        G_f2_25_q(i,j) = new_val_f2_25(15);  
                    elseif G_f2_25(i,j) >= edges(16) && G_f2_25(i,j) <= edges(17)   
                         G_f2_25_q(i,j) = new_val_f2_25(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_26,1) 
                for j = 1:size(G_f2_26,2) 
                    if G_f2_26(i,j) < edges(1)  
                        G_f2_26_q(i,j) = new_val_f2_26(1);  
                    elseif G_f2_26(i,j) >= edges(1) && G_f2_26(i,j) < edges(2)  
                        G_f2_26_q(i,j) = new_val_f2_26(1);  
                    elseif G_f2_26(i,j) >= edges(2) && G_f2_26(i,j) < edges(3)  
                        G_f2_26_q(i,j) = new_val_f2_26(2);    
                    elseif G_f2_26(i,j) >= edges(3) && G_f2_26(i,j) < edges(4)  
                        G_f2_26_q(i,j) = new_val_f2_26(3);  
                    elseif G_f2_26(i,j) >= edges(4) && G_f2_26(i,j) < edges(5)  
                        G_f2_26_q(i,j) = new_val_f2_26(4);   
                    elseif G_f2_26(i,j) >= edges(5) && G_f2_26(i,j) < edges(6)  
                        G_f2_26_q(i,j) = new_val_f2_26(5);     
                    elseif G_f2_26(i,j) >= edges(6) && G_f2_26(i,j) < edges(7)  
                        G_f2_26_q(i,j) = new_val_f2_26(6);  
                    elseif G_f2_26(i,j) >= edges(7) && G_f2_26(i,j) < edges(8)  
                        G_f2_26_q(i,j) = new_val_f2_26(7);   
                    elseif G_f2_26(i,j) >= edges(8) && G_f2_26(i,j) < edges(9)  
                        G_f2_26_q(i,j) = new_val_f2_26(8);     
                    elseif G_f2_26(i,j) >= edges(9) && G_f2_26(i,j) < edges(10)  
                        G_f2_26_q(i,j) = new_val_f2_26(9);  
                    elseif G_f2_26(i,j) >= edges(10) && G_f2_26(i,j) < edges(11)  
                        G_f2_26_q(i,j) = new_val_f2_26(10);  
                    elseif G_f2_26(i,j) >= edges(11) && G_f2_26(i,j) < edges(12)  
                        G_f2_26_q(i,j) = new_val_f2_26(11);  
                    elseif G_f2_26(i,j) >= edges(12) && G_f2_26(i,j) < edges(13)  
                        G_f2_26_q(i,j) = new_val_f2_26(12);  
                    elseif G_f2_26(i,j) >= edges(13) && G_f2_26(i,j) < edges(14)   
                         G_f2_26_q(i,j) = new_val_f2_26(13);   
                    elseif G_f2_26(i,j) >= edges(14) && G_f2_26(i,j) < edges(15)  
                        G_f2_26_q(i,j) = new_val_f2_26(14);  
                    elseif G_f2_26(i,j) >= edges(15) && G_f2_26(i,j) < edges(16)  
                        G_f2_26_q(i,j) = new_val_f2_26(15);  
                    elseif G_f2_26(i,j) >= edges(16) && G_f2_26(i,j) <= edges(17)   
                         G_f2_26_q(i,j) = new_val_f2_26(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_27,1) 
                for j = 1:size(G_f2_27,2) 
                    if G_f2_27(i,j) < edges(1)  
                        G_f2_27_q(i,j) = new_val_f2_27(1);  
                    elseif G_f2_27(i,j) >= edges(1) && G_f2_27(i,j) < edges(2)  
                        G_f2_27_q(i,j) = new_val_f2_27(1);  
                    elseif G_f2_27(i,j) >= edges(2) && G_f2_27(i,j) < edges(3)  
                        G_f2_27_q(i,j) = new_val_f2_27(2);    
                    elseif G_f2_27(i,j) >= edges(3) && G_f2_27(i,j) < edges(4)  
                        G_f2_27_q(i,j) = new_val_f2_27(3);  
                    elseif G_f2_27(i,j) >= edges(4) && G_f2_27(i,j) < edges(5)  
                        G_f2_27_q(i,j) = new_val_f2_27(4);   
                    elseif G_f2_27(i,j) >= edges(5) && G_f2_27(i,j) < edges(6)  
                        G_f2_27_q(i,j) = new_val_f2_27(5);     
                    elseif G_f2_27(i,j) >= edges(6) && G_f2_27(i,j) < edges(7)  
                        G_f2_27_q(i,j) = new_val_f2_27(6);  
                    elseif G_f2_27(i,j) >= edges(7) && G_f2_27(i,j) < edges(8)  
                        G_f2_27_q(i,j) = new_val_f2_27(7);   
                    elseif G_f2_27(i,j) >= edges(8) && G_f2_27(i,j) < edges(9)  
                        G_f2_27_q(i,j) = new_val_f2_27(8);     
                    elseif G_f2_27(i,j) >= edges(9) && G_f2_27(i,j) < edges(10)  
                        G_f2_27_q(i,j) = new_val_f2_27(9);  
                    elseif G_f2_27(i,j) >= edges(10) && G_f2_27(i,j) < edges(11)  
                        G_f2_27_q(i,j) = new_val_f2_27(10);  
                    elseif G_f2_27(i,j) >= edges(11) && G_f2_27(i,j) < edges(12)  
                        G_f2_27_q(i,j) = new_val_f2_27(11);  
                    elseif G_f2_27(i,j) >= edges(12) && G_f2_27(i,j) < edges(13)  
                        G_f2_27_q(i,j) = new_val_f2_27(12);  
                    elseif G_f2_27(i,j) >= edges(13) && G_f2_27(i,j) < edges(14)   
                         G_f2_27_q(i,j) = new_val_f2_27(13);   
                    elseif G_f2_27(i,j) >= edges(14) && G_f2_27(i,j) < edges(15)  
                        G_f2_27_q(i,j) = new_val_f2_27(14);  
                    elseif G_f2_27(i,j) >= edges(15) && G_f2_27(i,j) < edges(16)  
                        G_f2_27_q(i,j) = new_val_f2_27(15);  
                    elseif G_f2_27(i,j) >= edges(16) && G_f2_27(i,j) <= edges(17)   
                         G_f2_27_q(i,j) = new_val_f2_27(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_28,1) 
                for j = 1:size(G_f2_28,2) 
                    if G_f2_28(i,j) < edges(1)  
                        G_f2_28_q(i,j) = new_val_f2_28(1);  
                    elseif G_f2_28(i,j) >= edges(1) && G_f2_28(i,j) < edges(2)  
                        G_f2_28_q(i,j) = new_val_f2_28(1);  
                    elseif G_f2_28(i,j) >= edges(2) && G_f2_28(i,j) < edges(3)  
                        G_f2_28_q(i,j) = new_val_f2_28(2);    
                    elseif G_f2_28(i,j) >= edges(3) && G_f2_28(i,j) < edges(4)  
                        G_f2_28_q(i,j) = new_val_f2_28(3);  
                    elseif G_f2_28(i,j) >= edges(4) && G_f2_28(i,j) < edges(5)  
                        G_f2_28_q(i,j) = new_val_f2_28(4);   
                    elseif G_f2_28(i,j) >= edges(5) && G_f2_28(i,j) < edges(6)  
                        G_f2_28_q(i,j) = new_val_f2_28(5);     
                    elseif G_f2_28(i,j) >= edges(6) && G_f2_28(i,j) < edges(7)  
                        G_f2_28_q(i,j) = new_val_f2_28(6);  
                    elseif G_f2_28(i,j) >= edges(7) && G_f2_28(i,j) < edges(8)  
                        G_f2_28_q(i,j) = new_val_f2_28(7);   
                    elseif G_f2_28(i,j) >= edges(8) && G_f2_28(i,j) < edges(9)  
                        G_f2_28_q(i,j) = new_val_f2_28(8);     
                    elseif G_f2_28(i,j) >= edges(9) && G_f2_28(i,j) < edges(10)  
                        G_f2_28_q(i,j) = new_val_f2_28(9);  
                    elseif G_f2_28(i,j) >= edges(10) && G_f2_28(i,j) < edges(11)  
                        G_f2_28_q(i,j) = new_val_f2_28(10);  
                    elseif G_f2_28(i,j) >= edges(11) && G_f2_28(i,j) < edges(12)  
                        G_f2_28_q(i,j) = new_val_f2_28(11);  
                    elseif G_f2_28(i,j) >= edges(12) && G_f2_28(i,j) < edges(13)  
                        G_f2_28_q(i,j) = new_val_f2_28(12);  
                    elseif G_f2_28(i,j) >= edges(13) && G_f2_28(i,j) < edges(14)   
                         G_f2_28_q(i,j) = new_val_f2_28(13);   
                    elseif G_f2_28(i,j) >= edges(14) && G_f2_28(i,j) < edges(15)  
                        G_f2_28_q(i,j) = new_val_f2_28(14);  
                    elseif G_f2_28(i,j) >= edges(15) && G_f2_28(i,j) < edges(16)  
                        G_f2_28_q(i,j) = new_val_f2_28(15);  
                    elseif G_f2_28(i,j) >= edges(16) && G_f2_28(i,j) <= edges(17)   
                         G_f2_28_q(i,j) = new_val_f2_28(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_29,1) 
                for j = 1:size(G_f2_29,2) 
                    if G_f2_29(i,j) < edges(1)  
                        G_f2_29_q(i,j) = new_val_f2_29(1);  
                    elseif G_f2_29(i,j) >= edges(1) && G_f2_29(i,j) < edges(2)  
                        G_f2_29_q(i,j) = new_val_f2_29(1);  
                    elseif G_f2_29(i,j) >= edges(2) && G_f2_29(i,j) < edges(3)  
                        G_f2_29_q(i,j) = new_val_f2_29(2);    
                    elseif G_f2_29(i,j) >= edges(3) && G_f2_29(i,j) < edges(4)  
                        G_f2_29_q(i,j) = new_val_f2_29(3);  
                    elseif G_f2_29(i,j) >= edges(4) && G_f2_29(i,j) < edges(5)  
                        G_f2_29_q(i,j) = new_val_f2_29(4);   
                    elseif G_f2_29(i,j) >= edges(5) && G_f2_29(i,j) < edges(6)  
                        G_f2_29_q(i,j) = new_val_f2_29(5);     
                    elseif G_f2_29(i,j) >= edges(6) && G_f2_29(i,j) < edges(7)  
                        G_f2_29_q(i,j) = new_val_f2_29(6);  
                    elseif G_f2_29(i,j) >= edges(7) && G_f2_29(i,j) < edges(8)  
                        G_f2_29_q(i,j) = new_val_f2_29(7);   
                    elseif G_f2_29(i,j) >= edges(8) && G_f2_29(i,j) < edges(9)  
                        G_f2_29_q(i,j) = new_val_f2_29(8);     
                    elseif G_f2_29(i,j) >= edges(9) && G_f2_29(i,j) < edges(10)  
                        G_f2_29_q(i,j) = new_val_f2_29(9);  
                    elseif G_f2_29(i,j) >= edges(10) && G_f2_29(i,j) < edges(11)  
                        G_f2_29_q(i,j) = new_val_f2_29(10);  
                    elseif G_f2_29(i,j) >= edges(11) && G_f2_29(i,j) < edges(12)  
                        G_f2_29_q(i,j) = new_val_f2_29(11);  
                    elseif G_f2_29(i,j) >= edges(12) && G_f2_29(i,j) < edges(13)  
                        G_f2_29_q(i,j) = new_val_f2_29(12);  
                    elseif G_f2_29(i,j) >= edges(13) && G_f2_29(i,j) < edges(14)   
                         G_f2_29_q(i,j) = new_val_f2_29(13);   
                    elseif G_f2_29(i,j) >= edges(14) && G_f2_29(i,j) < edges(15)  
                        G_f2_29_q(i,j) = new_val_f2_29(14);  
                    elseif G_f2_29(i,j) >= edges(15) && G_f2_29(i,j) < edges(16)  
                        G_f2_29_q(i,j) = new_val_f2_29(15);  
                    elseif G_f2_29(i,j) >= edges(16) && G_f2_29(i,j) <= edges(17)   
                         G_f2_29_q(i,j) = new_val_f2_29(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_30,1) 
                for j = 1:size(G_f2_30,2) 
                    if G_f2_30(i,j) < edges(1)  
                        G_f2_30_q(i,j) = new_val_f2_30(1);  
                    elseif G_f2_30(i,j) >= edges(1) && G_f2_30(i,j) < edges(2)  
                        G_f2_30_q(i,j) = new_val_f2_30(1);  
                    elseif G_f2_30(i,j) >= edges(2) && G_f2_30(i,j) < edges(3)  
                        G_f2_30_q(i,j) = new_val_f2_30(2);    
                    elseif G_f2_30(i,j) >= edges(3) && G_f2_30(i,j) < edges(4)  
                        G_f2_30_q(i,j) = new_val_f2_30(3);  
                    elseif G_f2_30(i,j) >= edges(4) && G_f2_30(i,j) < edges(5)  
                        G_f2_30_q(i,j) = new_val_f2_30(4);   
                    elseif G_f2_30(i,j) >= edges(5) && G_f2_30(i,j) < edges(6)  
                        G_f2_30_q(i,j) = new_val_f2_30(5);     
                    elseif G_f2_30(i,j) >= edges(6) && G_f2_30(i,j) < edges(7)  
                        G_f2_30_q(i,j) = new_val_f2_30(6);  
                    elseif G_f2_30(i,j) >= edges(7) && G_f2_30(i,j) < edges(8)  
                        G_f2_30_q(i,j) = new_val_f2_30(7);   
                    elseif G_f2_30(i,j) >= edges(8) && G_f2_30(i,j) < edges(9)  
                        G_f2_30_q(i,j) = new_val_f2_30(8);     
                    elseif G_f2_30(i,j) >= edges(9) && G_f2_30(i,j) < edges(10)  
                        G_f2_30_q(i,j) = new_val_f2_30(9);  
                    elseif G_f2_30(i,j) >= edges(10) && G_f2_30(i,j) < edges(11)  
                        G_f2_30_q(i,j) = new_val_f2_30(10);  
                    elseif G_f2_30(i,j) >= edges(11) && G_f2_30(i,j) < edges(12)  
                        G_f2_30_q(i,j) = new_val_f2_30(11);  
                    elseif G_f2_30(i,j) >= edges(12) && G_f2_30(i,j) < edges(13)  
                        G_f2_30_q(i,j) = new_val_f2_30(12);  
                    elseif G_f2_30(i,j) >= edges(13) && G_f2_30(i,j) < edges(14)   
                         G_f2_30_q(i,j) = new_val_f2_30(13);   
                    elseif G_f2_30(i,j) >= edges(14) && G_f2_30(i,j) < edges(15)  
                        G_f2_30_q(i,j) = new_val_f2_30(14);  
                    elseif G_f2_30(i,j) >= edges(15) && G_f2_30(i,j) < edges(16)  
                        G_f2_30_q(i,j) = new_val_f2_30(15);  
                    elseif G_f2_30(i,j) >= edges(16) && G_f2_30(i,j) <= edges(17)   
                         G_f2_30_q(i,j) = new_val_f2_30(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_31,1) 
                for j = 1:size(G_f2_31,2) 
                    if G_f2_31(i,j) < edges(1)  
                        G_f2_31_q(i,j) = new_val_f2_31(1);  
                    elseif G_f2_31(i,j) >= edges(1) && G_f2_31(i,j) < edges(2)  
                        G_f2_31_q(i,j) = new_val_f2_31(1);  
                    elseif G_f2_31(i,j) >= edges(2) && G_f2_31(i,j) < edges(3)  
                        G_f2_31_q(i,j) = new_val_f2_31(2);    
                    elseif G_f2_31(i,j) >= edges(3) && G_f2_31(i,j) < edges(4)  
                        G_f2_31_q(i,j) = new_val_f2_31(3);  
                    elseif G_f2_31(i,j) >= edges(4) && G_f2_31(i,j) < edges(5)  
                        G_f2_31_q(i,j) = new_val_f2_31(4);   
                    elseif G_f2_31(i,j) >= edges(5) && G_f2_31(i,j) < edges(6)  
                        G_f2_31_q(i,j) = new_val_f2_31(5);     
                    elseif G_f2_31(i,j) >= edges(6) && G_f2_31(i,j) < edges(7)  
                        G_f2_31_q(i,j) = new_val_f2_31(6);  
                    elseif G_f2_31(i,j) >= edges(7) && G_f2_31(i,j) < edges(8)  
                        G_f2_31_q(i,j) = new_val_f2_31(7);   
                    elseif G_f2_31(i,j) >= edges(8) && G_f2_31(i,j) < edges(9)  
                        G_f2_31_q(i,j) = new_val_f2_31(8);     
                    elseif G_f2_31(i,j) >= edges(9) && G_f2_31(i,j) < edges(10)  
                        G_f2_31_q(i,j) = new_val_f2_31(9);  
                    elseif G_f2_31(i,j) >= edges(10) && G_f2_31(i,j) < edges(11)  
                        G_f2_31_q(i,j) = new_val_f2_31(10);  
                    elseif G_f2_31(i,j) >= edges(11) && G_f2_31(i,j) < edges(12)  
                        G_f2_31_q(i,j) = new_val_f2_31(11);  
                    elseif G_f2_31(i,j) >= edges(12) && G_f2_31(i,j) < edges(13)  
                        G_f2_31_q(i,j) = new_val_f2_31(12);  
                    elseif G_f2_31(i,j) >= edges(13) && G_f2_31(i,j) < edges(14)   
                         G_f2_31_q(i,j) = new_val_f2_31(13);   
                    elseif G_f2_31(i,j) >= edges(14) && G_f2_31(i,j) < edges(15)  
                        G_f2_31_q(i,j) = new_val_f2_31(14);  
                    elseif G_f2_31(i,j) >= edges(15) && G_f2_31(i,j) < edges(16)  
                        G_f2_31_q(i,j) = new_val_f2_31(15);  
                    elseif G_f2_31(i,j) >= edges(16) && G_f2_31(i,j) <= edges(17)   
                         G_f2_31_q(i,j) = new_val_f2_31(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f2_32,1) 
                for j = 1:size(G_f2_32,2) 
                    if G_f2_32(i,j) < edges(1)  
                        G_f2_32_q(i,j) = new_val_f2_32(1);  
                    elseif G_f2_32(i,j) >= edges(1) && G_f2_32(i,j) < edges(2)  
                        G_f2_32_q(i,j) = new_val_f2_32(1);  
                    elseif G_f2_32(i,j) >= edges(2) && G_f2_32(i,j) < edges(3)  
                        G_f2_32_q(i,j) = new_val_f2_32(2);    
                    elseif G_f2_32(i,j) >= edges(3) && G_f2_32(i,j) < edges(4)  
                        G_f2_32_q(i,j) = new_val_f2_32(3);  
                    elseif G_f2_32(i,j) >= edges(4) && G_f2_32(i,j) < edges(5)  
                        G_f2_32_q(i,j) = new_val_f2_32(4);   
                    elseif G_f2_32(i,j) >= edges(5) && G_f2_32(i,j) < edges(6)  
                        G_f2_32_q(i,j) = new_val_f2_32(5);     
                    elseif G_f2_32(i,j) >= edges(6) && G_f2_32(i,j) < edges(7)  
                        G_f2_32_q(i,j) = new_val_f2_32(6);  
                    elseif G_f2_32(i,j) >= edges(7) && G_f2_32(i,j) < edges(8)  
                        G_f2_32_q(i,j) = new_val_f2_32(7);   
                    elseif G_f2_32(i,j) >= edges(8) && G_f2_32(i,j) < edges(9)  
                        G_f2_32_q(i,j) = new_val_f2_32(8);     
                    elseif G_f2_32(i,j) >= edges(9) && G_f2_32(i,j) < edges(10)  
                        G_f2_32_q(i,j) = new_val_f2_32(9);  
                    elseif G_f2_32(i,j) >= edges(10) && G_f2_32(i,j) < edges(11)  
                        G_f2_32_q(i,j) = new_val_f2_32(10);  
                    elseif G_f2_32(i,j) >= edges(11) && G_f2_32(i,j) < edges(12)  
                        G_f2_32_q(i,j) = new_val_f2_32(11);  
                    elseif G_f2_32(i,j) >= edges(12) && G_f2_32(i,j) < edges(13)  
                        G_f2_32_q(i,j) = new_val_f2_32(12);  
                    elseif G_f2_32(i,j) >= edges(13) && G_f2_32(i,j) < edges(14)   
                         G_f2_32_q(i,j) = new_val_f2_32(13);   
                    elseif G_f2_32(i,j) >= edges(14) && G_f2_32(i,j) < edges(15)  
                        G_f2_32_q(i,j) = new_val_f2_32(14);  
                    elseif G_f2_32(i,j) >= edges(15) && G_f2_32(i,j) < edges(16)  
                        G_f2_32_q(i,j) = new_val_f2_32(15);  
                    elseif G_f2_32(i,j) >= edges(16) && G_f2_32(i,j) <= edges(17)   
                         G_f2_32_q(i,j) = new_val_f2_32(16);    
                    end 
                end 
            end 
                        
        
        
        out_1_2_cond = im_cond_1*G_f2_1_q; 
        out_2_2_cond = im_cond_2*G_f2_2_q; 
        out_3_2_cond = im_cond_3*G_f2_3_q; 
        out_4_2_cond = im_cond_4*G_f2_4_q; 
        out_5_2_cond = im_cond_5*G_f2_5_q; 
        out_6_2_cond = im_cond_6*G_f2_6_q; 
        out_7_2_cond = im_cond_7*G_f2_7_q; 
        out_8_2_cond = im_cond_8*G_f2_8_q; 
        out_9_2_cond = im_cond_9*G_f2_9_q; 
        out_10_2_cond = im_cond_10*G_f2_10_q; 
        out_11_2_cond = im_cond_11*G_f2_11_q; 
        out_12_2_cond = im_cond_12*G_f2_12_q; 
        out_13_2_cond = im_cond_13*G_f2_13_q; 
        out_14_2_cond = im_cond_14*G_f2_14_q; 
        out_15_2_cond = im_cond_15*G_f2_15_q; 
        out_16_2_cond = im_cond_16*G_f2_16_q; 
        out_17_2_cond = im_cond_17*G_f2_17_q; 
        out_18_2_cond = im_cond_18*G_f2_18_q; 
        out_19_2_cond = im_cond_19*G_f2_19_q; 
        out_20_2_cond = im_cond_20*G_f2_20_q; 
        out_21_2_cond = im_cond_21*G_f2_21_q; 
        out_22_2_cond = im_cond_22*G_f2_22_q; 
        out_23_2_cond = im_cond_23*G_f2_23_q; 
        out_24_2_cond = im_cond_24*G_f2_24_q; 
        out_25_2_cond = im_cond_25*G_f2_25_q; 
        out_26_2_cond = im_cond_26*G_f2_26_q; 
        out_27_2_cond = im_cond_27*G_f2_27_q; 
        out_28_2_cond = im_cond_28*G_f2_28_q; 
        out_29_2_cond = im_cond_29*G_f2_29_q; 
        out_30_2_cond = im_cond_30*G_f2_30_q; 
        out_31_2_cond = im_cond_31*G_f2_31_q; 
        out_32_2_cond = im_cond_32*G_f2_32_q; 

        output_second_layer_with_shifting_cond = out_1_2_cond + out_2_2_cond +...
            out_3_2_cond + out_4_2_cond + out_5_2_cond + out_6_2_cond + out_7_2_cond +...
            out_8_2_cond + out_9_2_cond + out_10_2_cond + out_11_2_cond + out_12_2_cond +...
            out_13_2_cond + out_14_2_cond + out_15_2_cond + out_16_2_cond + out_17_2_cond +...
            out_18_2_cond + out_19_2_cond + out_20_2_cond + out_21_2_cond + out_22_2_cond +...
            out_23_2_cond + out_24_2_cond + out_25_2_cond + out_26_2_cond + out_27_2_cond +...
            out_28_2_cond + out_29_2_cond + out_30_2_cond + out_31_2_cond + out_32_2_cond ;

        shifting_matrix_cond2 = output_second_layer_with_shifting_cond(:,33:end);
        output_second_layer_with_shifting_cond = output_second_layer_with_shifting_cond(:,1:32);

        % Each col is an output filter (its flattening happened row-by-row)
        output_second_layer_with_shifting_cond = output_second_layer_with_shifting_cond - shifting_matrix_cond2;

        % Re-shape it to pass it through pooling and relu
        output_2_cond = permute(reshape(output_second_layer_with_shifting_cond,[16,16,32]),[2 1 3]);


        % ReLU 2
        for i =1: size(output_2_shifting,1)
            for j = 1: size(output_2_shifting,2)
                for k = 1: size(output_2_shifting,3)
                    if output_2_shifting(i,j,k) < 0
                        output_2_shifting(i,j,k) = 0;
                    end
                end
            end
        end


        for i =1: size(output_2_cond,1)
            for j = 1: size(output_2_cond,2)
                for k = 1: size(output_2_cond,3)
                    if output_2_cond(i,j,k) < 0
                        output_2_cond(i,j,k) = 0;
                    end
                end
            end
        end


        % Average Pooling
        PoolMap2_shifting = zeros(size(pool2,1),size(pool2,2), size(pool2,3));
        row1 = 1;
        col1 = 3;
        row2 = 1;
        col2 = 3;
        k = 3; % kernel size
        for j = 1 : numOfFilters2 %across all filters
            k1 = 1;
            for s1 = 0:2:(size(output_2_shifting,1)-2) %controls the shifting of the row and col per col
                k2 = 1;
                for s2 = 0:2:(size(output_2_shifting,1)-2) %controls the shifting of the col
                    if (s1 == (size(output_2_shifting,1)-2)) && (s2 == (size(output_2_shifting,1)-2))
                        out1 = sum(sum(output_2_shifting(row1+s1:col1+s1-1,row2+s2:col2+s2-1,j)))/((k-1)*(k-1));
                    elseif s2 == (size(output_2_shifting,1)-2)
                        out1 = sum(sum(output_2_shifting(row1+s1:col1+s1,row2+s2:col2+s2-1,j)))/((k-1)*k); 
                    elseif s1 == (size(output_2_shifting,1)-2)
                        out1 = sum(sum(output_2_shifting(row1+s1:col1+s1-1,row2+s2:col2+s2,j)))/((k-1)*k); 
                    else 
                        out1 = sum(sum(output_2_shifting(row1+s1:col1+s1,row2+s2:col2+s2,j)))/(k*k);  
                    end    
                    PoolMap2_shifting(k1,k2,j) = out1;
                    k2 = k2 + 1;

                end
                k1 = k1 + 1;
            end  
        end


        % Average Pooling
        PoolMap2_shifting_cond = zeros(size(pool2,1),size(pool2,2), size(pool2,3));
        row1 = 1;
        col1 = 3;
        row2 = 1;
        col2 = 3;
        k = 3; % kernel size
        for j = 1 : numOfFilters2 %across all filters
            k1 = 1;
            for s1 = 0:2:(size(output_2_cond,1)-2) %controls the shifting of the row and col per col
                k2 = 1;
                for s2 = 0:2:(size(output_2_cond,1)-2) %controls the shifting of the col
                    if (s1 == (size(output_2_cond,1)-2)) && (s2 == (size(output_2_cond,1)-2))
                        out1 = sum(sum(output_2_cond(row1+s1:col1+s1-1,row2+s2:col2+s2-1,j)))/((k-1)*(k-1));
                    elseif s2 == (size(output_2_cond,1)-2)
                        out1 = sum(sum(output_2_cond(row1+s1:col1+s1,row2+s2:col2+s2-1,j)))/((k-1)*k); 
                    elseif s1 == (size(output_2_cond,1)-2)
                        out1 = sum(sum(output_2_cond(row1+s1:col1+s1-1,row2+s2:col2+s2,j)))/((k-1)*k); 
                    else 
                        out1 = sum(sum(output_2_cond(row1+s1:col1+s1,row2+s2:col2+s2,j)))/(k*k);  
                    end    
                    PoolMap2_shifting_cond(k1,k2,j) = out1;
                    k2 = k2 + 1;

                end
                k1 = k1 + 1;
            end  
        end

        offset_2 = 25.4192;
        PoolMap2_shifting_cond = PoolMap2_shifting_cond * offset_2;
        
        
        f3_1 = zeros(25,32); 
        f3_2 = zeros(25,32); 
        f3_3 = zeros(25,32); 
        f3_4 = zeros(25,32); 
        f3_5 = zeros(25,32); 
        f3_6 = zeros(25,32); 
        f3_7 = zeros(25,32); 
        f3_8 = zeros(25,32); 
        f3_9 = zeros(25,32); 
        f3_10 = zeros(25,32); 
        f3_11 = zeros(25,32); 
        f3_12 = zeros(25,32); 
        f3_13 = zeros(25,32); 
        f3_14 = zeros(25,32); 
        f3_15 = zeros(25,32); 
        f3_16 = zeros(25,32); 
        f3_17 = zeros(25,32); 
        f3_18 = zeros(25,32); 
        f3_19 = zeros(25,32); 
        f3_20 = zeros(25,32); 
        f3_21 = zeros(25,32); 
        f3_22 = zeros(25,32); 
        f3_23 = zeros(25,32); 
        f3_24 = zeros(25,32); 
        f3_25 = zeros(25,32); 
        f3_26 = zeros(25,32); 
        f3_27 = zeros(25,32); 
        f3_28 = zeros(25,32); 
        f3_29 = zeros(25,32); 
        f3_30 = zeros(25,32); 
        f3_31 = zeros(25,32); 
        f3_32 = zeros(25,32); 


        % data_padded3_shift = padarray(PoolMap2,[pad3 pad3],0,'both'); 
        data_padded3_shift = padarray(PoolMap2_shifting,[pad3 pad3],0,'both');
        data_padded3_shift_cond = padarray(PoolMap2_shifting_cond,[pad3 pad3],0,'both');

        for j = 1 : numOfFilters3 
            f3_1(:,j) = reshape(w3(:,:,1,j), 1,25);
            f3_2(:,j) = reshape(w3(:,:,2,j), 1,25);
            f3_3(:,j) = reshape(w3(:,:,3,j), 1,25);
            f3_4(:,j) = reshape(w3(:,:,4,j), 1,25);
            f3_5(:,j) = reshape(w3(:,:,5,j), 1,25);
            f3_6(:,j) = reshape(w3(:,:,6,j), 1,25);
            f3_7(:,j) = reshape(w3(:,:,7,j), 1,25);
            f3_8(:,j) = reshape(w3(:,:,8,j), 1,25);
            f3_9(:,j) = reshape(w3(:,:,9,j), 1,25);
            f3_10(:,j) = reshape(w3(:,:,10,j), 1,25);
            f3_11(:,j) = reshape(w3(:,:,11,j), 1,25);
            f3_12(:,j) = reshape(w3(:,:,12,j), 1,25);
            f3_13(:,j) = reshape(w3(:,:,13,j), 1,25);
            f3_14(:,j) = reshape(w3(:,:,14,j), 1,25);
            f3_15(:,j) = reshape(w3(:,:,15,j), 1,25);
            f3_16(:,j) = reshape(w3(:,:,16,j), 1,25);
            f3_17(:,j) = reshape(w3(:,:,17,j), 1,25);
            f3_18(:,j) = reshape(w3(:,:,18,j), 1,25);
            f3_19(:,j) = reshape(w3(:,:,19,j), 1,25);
            f3_20(:,j) = reshape(w3(:,:,20,j), 1,25);
            f3_21(:,j) = reshape(w3(:,:,21,j), 1,25);
            f3_22(:,j) = reshape(w3(:,:,22,j), 1,25);
            f3_23(:,j) = reshape(w3(:,:,23,j), 1,25);
            f3_24(:,j) = reshape(w3(:,:,24,j), 1,25);
            f3_25(:,j) = reshape(w3(:,:,25,j), 1,25);
            f3_26(:,j) = reshape(w3(:,:,26,j), 1,25);
            f3_27(:,j) = reshape(w3(:,:,27,j), 1,25);
            f3_28(:,j) = reshape(w3(:,:,28,j), 1,25);
            f3_29(:,j) = reshape(w3(:,:,29,j), 1,25);
            f3_30(:,j) = reshape(w3(:,:,30,j), 1,25);
            f3_31(:,j) = reshape(w3(:,:,31,j), 1,25);
            f3_32(:,j) = reshape(w3(:,:,32,j), 1,25);

        end

        f3_1 =  [f3_1; b3'];    

        im_3_1 = cell(1,1);
        im_3_2 = cell(1,1);
        im_3_3 = cell(1,1);
        im_3_4 = cell(1,1);
        im_3_5 = cell(1,1);
        im_3_6 = cell(1,1);
        im_3_7 = cell(1,1);
        im_3_8 = cell(1,1);
        im_3_9 = cell(1,1);
        im_3_10 = cell(1,1);
        im_3_11 = cell(1,1);
        im_3_12 = cell(1,1);
        im_3_13 = cell(1,1);
        im_3_14 = cell(1,1);
        im_3_15 = cell(1,1);
        im_3_16 = cell(1,1);
        im_3_17 = cell(1,1);
        im_3_18 = cell(1,1);
        im_3_19 = cell(1,1);
        im_3_20 = cell(1,1);
        im_3_21 = cell(1,1);
        im_3_22 = cell(1,1);
        im_3_23 = cell(1,1);
        im_3_24 = cell(1,1);
        im_3_25 = cell(1,1);
        im_3_26 = cell(1,1);
        im_3_27 = cell(1,1);
        im_3_28 = cell(1,1);
        im_3_29 = cell(1,1);
        im_3_30 = cell(1,1);
        im_3_31 = cell(1,1);
        im_3_32 = cell(1,1);

        im_cond_3_1 = cell(1,1);
        im_cond_3_2 = cell(1,1);
        im_cond_3_3 = cell(1,1);
        im_cond_3_4 = cell(1,1);
        im_cond_3_5 = cell(1,1);
        im_cond_3_6 = cell(1,1);
        im_cond_3_7 = cell(1,1);
        im_cond_3_8 = cell(1,1);
        im_cond_3_9 = cell(1,1);
        im_cond_3_10 = cell(1,1);
        im_cond_3_11 = cell(1,1);
        im_cond_3_12 = cell(1,1);
        im_cond_3_13 = cell(1,1);
        im_cond_3_14 = cell(1,1);
        im_cond_3_15 = cell(1,1);
        im_cond_3_16 = cell(1,1);
        im_cond_3_17 = cell(1,1);
        im_cond_3_18 = cell(1,1);
        im_cond_3_19 = cell(1,1);
        im_cond_3_20 = cell(1,1);
        im_cond_3_21 = cell(1,1);
        im_cond_3_22 = cell(1,1);
        im_cond_3_23 = cell(1,1);
        im_cond_3_24 = cell(1,1);
        im_cond_3_25 = cell(1,1);
        im_cond_3_26 = cell(1,1);
        im_cond_3_27 = cell(1,1);
        im_cond_3_28 = cell(1,1);
        im_cond_3_29 = cell(1,1);
        im_cond_3_30 = cell(1,1);
        im_cond_3_31 = cell(1,1);
        im_cond_3_32 = cell(1,1);



        row1 = 1;
        col1 = 5;
        row2 = 1;
        col2 = 5;

        for i = 1: numOfchannels_perFilter3 %across all channels of a filter
            counter = 1;
            for s1 = 0:(size(data_padded3_shift,1)-col1) %controls the shifting of the row and col per col
                for s2 = 0:(size(data_padded3_shift,1)-col1) %controls the shifting of the col
                    input = data_padded3_shift(row1+s1:col1+s1,row2+s2:col2+s2,i);
                    input = reshape(input,1,25);
                    if i == 1
                       im_3_1{counter,1} = input;
                    elseif i == 2
                       im_3_2{counter,1} = input;
                    elseif i == 3
                       im_3_3{counter,1} = input;
                    elseif i == 4
                       im_3_4{counter,1} = input;
                    elseif i == 5
                       im_3_5{counter,1} = input;                
                    elseif i == 6
                       im_3_6{counter,1} = input;
                    elseif i == 7
                       im_3_7{counter,1} = input;                
                    elseif i == 8
                       im_3_8{counter,1} = input;
                    elseif i == 9
                       im_3_9{counter,1} = input;                
                    elseif i == 10
                       im_3_10{counter,1} = input;
                    elseif i == 11
                       im_3_11{counter,1} = input;                
                    elseif i == 12
                       im_3_12{counter,1} = input;
                    elseif i == 13
                       im_3_13{counter,1} = input;                
                    elseif i == 14
                       im_3_14{counter,1} = input;
                    elseif i == 15
                       im_3_15{counter,1} = input;                
                    elseif i == 16
                       im_3_16{counter,1} = input;
                    elseif i == 17
                       im_3_17{counter,1} = input;                
                    elseif i == 18
                       im_3_18{counter,1} = input;
                    elseif i == 19
                       im_3_19{counter,1} = input;                
                    elseif i == 20
                       im_3_20{counter,1} = input;
                    elseif i == 21
                       im_3_21{counter,1} = input;                
                    elseif i == 22
                       im_3_22{counter,1} = input;
                    elseif i == 23
                       im_3_23{counter,1} = input;
                     elseif i == 24
                       im_3_24{counter,1} = input;
                    elseif i == 25
                       im_3_25{counter,1} = input;                
                    elseif i == 26
                       im_3_26{counter,1} = input;
                    elseif i == 27
                       im_3_27{counter,1} = input;                
                    elseif i == 28
                       im_3_28{counter,1} = input;
                    elseif i == 29
                       im_3_29{counter,1} = input;                
                    elseif i == 30
                       im_3_30{counter,1} = input;
                    elseif i == 31
                       im_3_31{counter,1} = input;                                                        
                    else
                       im_3_32{counter,1} = input;
                    end
                    counter = counter + 1;
                end
            end
        end  

        %Each image is a row and each filter is a column
        im_3_1 = cell2mat(im_3_1);
        im_3_1 = [im_3_1,ones(length(im_3_1),1)];

        im_3_2 = cell2mat(im_3_2);
        im_3_3 = cell2mat(im_3_3);
        im_3_4 = cell2mat(im_3_4);
        im_3_5 = cell2mat(im_3_5);
        im_3_6 = cell2mat(im_3_6);
        im_3_7 = cell2mat(im_3_7);
        im_3_8 = cell2mat(im_3_8);
        im_3_9 = cell2mat(im_3_9);
        im_3_10 = cell2mat(im_3_10);
        im_3_11 = cell2mat(im_3_11);
        im_3_12 = cell2mat(im_3_12);
        im_3_13 = cell2mat(im_3_13);
        im_3_14 = cell2mat(im_3_14);
        im_3_15 = cell2mat(im_3_15);
        im_3_16 = cell2mat(im_3_16);
        im_3_17 = cell2mat(im_3_17);
        im_3_18 = cell2mat(im_3_18);
        im_3_19 = cell2mat(im_3_19);
        im_3_20 = cell2mat(im_3_20);
        im_3_21 = cell2mat(im_3_21);
        im_3_22 = cell2mat(im_3_22);
        im_3_23 = cell2mat(im_3_23);
        im_3_24 = cell2mat(im_3_24);
        im_3_25 = cell2mat(im_3_25);
        im_3_26 = cell2mat(im_3_26);
        im_3_27 = cell2mat(im_3_27);
        im_3_28 = cell2mat(im_3_28);
        im_3_29 = cell2mat(im_3_29);
        im_3_30 = cell2mat(im_3_30);
        im_3_31 = cell2mat(im_3_31);
        im_3_32 = cell2mat(im_3_32);


        for i = 1: numOfchannels_perFilter3 %across all channels of a filter
            counter = 1;
            for s1 = 0:(size(data_padded3_shift_cond,1)-col1) %controls the shifting of the row and col per col
                for s2 = 0:(size(data_padded3_shift_cond,1)-col1) %controls the shifting of the col
                    input = data_padded3_shift_cond(row1+s1:col1+s1,row2+s2:col2+s2,i);
                    input = reshape(input,1,25);
                    if i == 1
                      im_cond_3_1{counter,1} = input;
                    elseif i == 2
                      im_cond_3_2{counter,1} = input;
                    elseif i == 3
                      im_cond_3_3{counter,1} = input;
                    elseif i == 4
                      im_cond_3_4{counter,1} = input;
                    elseif i == 5
                      im_cond_3_5{counter,1} = input;                
                    elseif i == 6
                      im_cond_3_6{counter,1} = input;
                    elseif i == 7
                      im_cond_3_7{counter,1} = input;                
                    elseif i == 8
                      im_cond_3_8{counter,1} = input;
                    elseif i == 9
                      im_cond_3_9{counter,1} = input;                
                    elseif i == 10
                      im_cond_3_10{counter,1} = input;
                    elseif i == 11
                      im_cond_3_11{counter,1} = input;                
                    elseif i == 12
                      im_cond_3_12{counter,1} = input;
                    elseif i == 13
                      im_cond_3_13{counter,1} = input;                
                    elseif i == 14
                      im_cond_3_14{counter,1} = input;
                    elseif i == 15
                      im_cond_3_15{counter,1} = input;                
                    elseif i == 16
                      im_cond_3_16{counter,1} = input;
                    elseif i == 17
                      im_cond_3_17{counter,1} = input;                
                    elseif i == 18
                      im_cond_3_18{counter,1} = input;
                    elseif i == 19
                      im_cond_3_19{counter,1} = input;                
                    elseif i == 20
                      im_cond_3_20{counter,1} = input;
                    elseif i == 21
                      im_cond_3_21{counter,1} = input;                
                    elseif i == 22
                      im_cond_3_22{counter,1} = input;
                    elseif i == 23
                      im_cond_3_23{counter,1} = input;
                     elseif i == 24
                      im_cond_3_24{counter,1} = input;
                    elseif i == 25
                      im_cond_3_25{counter,1} = input;                
                    elseif i == 26
                      im_cond_3_26{counter,1} = input;
                    elseif i == 27
                      im_cond_3_27{counter,1} = input;                
                    elseif i == 28
                      im_cond_3_28{counter,1} = input;
                    elseif i == 29
                      im_cond_3_29{counter,1} = input;                
                    elseif i == 30
                      im_cond_3_30{counter,1} = input;
                    elseif i == 31
                      im_cond_3_31{counter,1} = input;                                                        
                    else
                      im_cond_3_32{counter,1} = input;
                    end
                    counter = counter + 1;
                end
            end
        end  

        %Each image is a row and each filter is a column
        im_cond_3_1 = cell2mat(im_cond_3_1);
        im_cond_3_1 = [im_cond_3_1,ones(length(im_cond_3_1),1)];

        im_cond_3_2 = cell2mat(im_cond_3_2);
        im_cond_3_3 = cell2mat(im_cond_3_3);
        im_cond_3_4 = cell2mat(im_cond_3_4);
        im_cond_3_5 = cell2mat(im_cond_3_5);
        im_cond_3_6 = cell2mat(im_cond_3_6);
        im_cond_3_7 = cell2mat(im_cond_3_7);
        im_cond_3_8 = cell2mat(im_cond_3_8);
        im_cond_3_9 = cell2mat(im_cond_3_9);
        im_cond_3_10 = cell2mat(im_cond_3_10);
        im_cond_3_11 = cell2mat(im_cond_3_11);
        im_cond_3_12 = cell2mat(im_cond_3_12);
        im_cond_3_13 = cell2mat(im_cond_3_13);
        im_cond_3_14 = cell2mat(im_cond_3_14);
        im_cond_3_15 = cell2mat(im_cond_3_15);
        im_cond_3_16 = cell2mat(im_cond_3_16);
        im_cond_3_17 = cell2mat(im_cond_3_17);
        im_cond_3_18 = cell2mat(im_cond_3_18);
        im_cond_3_19 = cell2mat(im_cond_3_19);
        im_cond_3_20 = cell2mat(im_cond_3_20);
        im_cond_3_21 = cell2mat(im_cond_3_21);
        im_cond_3_22 = cell2mat(im_cond_3_22);
        im_cond_3_23 = cell2mat(im_cond_3_23);
        im_cond_3_24 = cell2mat(im_cond_3_24);
        im_cond_3_25 = cell2mat(im_cond_3_25);
        im_cond_3_26 = cell2mat(im_cond_3_26);
        im_cond_3_27 = cell2mat(im_cond_3_27);
        im_cond_3_28 = cell2mat(im_cond_3_28);
        im_cond_3_29 = cell2mat(im_cond_3_29);
        im_cond_3_30 = cell2mat(im_cond_3_30);
        im_cond_3_31 = cell2mat(im_cond_3_31);
        im_cond_3_32 = cell2mat(im_cond_3_32);


        shift_3_1 = zeros(numOfFilters3,1);
        shift_3_2 = zeros(numOfFilters3,1);
        shift_3_3 = zeros(numOfFilters3,1);
        shift_3_4 = zeros(numOfFilters3,1);
        shift_3_5 = zeros(numOfFilters3,1);
        shift_3_6 = zeros(numOfFilters3,1);
        shift_3_7 = zeros(numOfFilters3,1);
        shift_3_8 = zeros(numOfFilters3,1);
        shift_3_9 = zeros(numOfFilters3,1);
        shift_3_10 = zeros(numOfFilters3,1);
        shift_3_11 = zeros(numOfFilters3,1);
        shift_3_12 = zeros(numOfFilters3,1);
        shift_3_13 = zeros(numOfFilters3,1);
        shift_3_14 = zeros(numOfFilters3,1);
        shift_3_15 = zeros(numOfFilters3,1);
        shift_3_16 = zeros(numOfFilters3,1);
        shift_3_17 = zeros(numOfFilters3,1);
        shift_3_18 = zeros(numOfFilters3,1);
        shift_3_19 = zeros(numOfFilters3,1);
        shift_3_20 = zeros(numOfFilters3,1);
        shift_3_21 = zeros(numOfFilters3,1);
        shift_3_22 = zeros(numOfFilters3,1);
        shift_3_23 = zeros(numOfFilters3,1);
        shift_3_24 = zeros(numOfFilters3,1);
        shift_3_25 = zeros(numOfFilters3,1);
        shift_3_26 = zeros(numOfFilters3,1);
        shift_3_27 = zeros(numOfFilters3,1);
        shift_3_28 = zeros(numOfFilters3,1);
        shift_3_29 = zeros(numOfFilters3,1);
        shift_3_30 = zeros(numOfFilters3,1);
        shift_3_31 = zeros(numOfFilters3,1);
        shift_3_32 = zeros(numOfFilters3,1);

        for j = 1 : numOfFilters3 %across all filters
            shift_3_1(j) = min(f3_1(:,j));
            shift_3_2(j) = min(f3_2(:,j));
            shift_3_3(j) = min(f3_3(:,j));
            shift_3_4(j) = min(f3_4(:,j));
            shift_3_5(j) = min(f3_5(:,j));
            shift_3_6(j) = min(f3_6(:,j));
            shift_3_7(j) = min(f3_7(:,j));
            shift_3_8(j) = min(f3_8(:,j));
            shift_3_9(j) = min(f3_9(:,j));
            shift_3_10(j) = min(f3_10(:,j));
            shift_3_11(j) = min(f3_11(:,j));
            shift_3_12(j) = min(f3_12(:,j));
            shift_3_13(j) = min(f3_13(:,j));
            shift_3_14(j) = min(f3_14(:,j));
            shift_3_15(j) = min(f3_15(:,j));
            shift_3_16(j) = min(f3_16(:,j));
            shift_3_17(j) = min(f3_17(:,j));
            shift_3_18(j) = min(f3_18(:,j));
            shift_3_19(j) = min(f3_19(:,j));
            shift_3_20(j) = min(f3_20(:,j));
            shift_3_21(j) = min(f3_21(:,j));
            shift_3_22(j) = min(f3_22(:,j));
            shift_3_23(j) = min(f3_23(:,j));
            shift_3_24(j) = min(f3_24(:,j));
            shift_3_25(j) = min(f3_25(:,j));
            shift_3_26(j) = min(f3_26(:,j));
            shift_3_27(j) = min(f3_27(:,j));
            shift_3_28(j) = min(f3_28(:,j));
            shift_3_29(j) = min(f3_29(:,j));
            shift_3_30(j) = min(f3_30(:,j));
            shift_3_31(j) = min(f3_31(:,j));
            shift_3_32(j) = min(f3_32(:,j));
        end

        for j = 1 : numOfFilters3 %across all filters
            f3_1(:,j) = f3_1(:,j) -  shift_3_1(j);
            f3_2(:,j) = f3_2(:,j) -  shift_3_2(j);
            f3_3(:,j) = f3_3(:,j) -  shift_3_3(j);
            f3_4(:,j) = f3_4(:,j) -  shift_3_4(j);
            f3_5(:,j) = f3_5(:,j) -  shift_3_5(j);
            f3_6(:,j) = f3_6(:,j) -  shift_3_6(j);
            f3_7(:,j) = f3_7(:,j) -  shift_3_7(j);
            f3_8(:,j) = f3_8(:,j) -  shift_3_8(j);
            f3_9(:,j) = f3_9(:,j) -  shift_3_9(j);
            f3_10(:,j) = f3_10(:,j) -  shift_3_10(j);
            f3_11(:,j) = f3_11(:,j) -  shift_3_11(j);
            f3_12(:,j) = f3_12(:,j) -  shift_3_12(j);
            f3_13(:,j) = f3_13(:,j) -  shift_3_13(j);
            f3_14(:,j) = f3_14(:,j) -  shift_3_14(j);
            f3_15(:,j) = f3_15(:,j) -  shift_3_15(j);
            f3_16(:,j) = f3_16(:,j) -  shift_3_16(j);
            f3_17(:,j) = f3_17(:,j) -  shift_3_17(j);
            f3_18(:,j) = f3_18(:,j) -  shift_3_18(j);
            f3_19(:,j) = f3_19(:,j) -  shift_3_19(j);
            f3_20(:,j) = f3_20(:,j) -  shift_3_20(j);
            f3_21(:,j) = f3_21(:,j) -  shift_3_21(j);
            f3_22(:,j) = f3_22(:,j) -  shift_3_22(j);
            f3_23(:,j) = f3_23(:,j) -  shift_3_23(j);
            f3_24(:,j) = f3_24(:,j) -  shift_3_24(j);
            f3_25(:,j) = f3_25(:,j) -  shift_3_25(j);
            f3_26(:,j) = f3_26(:,j) -  shift_3_26(j);
            f3_27(:,j) = f3_27(:,j) -  shift_3_27(j);
            f3_28(:,j) = f3_28(:,j) -  shift_3_28(j);
            f3_29(:,j) = f3_29(:,j) -  shift_3_29(j);
            f3_30(:,j) = f3_30(:,j) -  shift_3_30(j);
            f3_31(:,j) = f3_31(:,j) -  shift_3_31(j);
            f3_32(:,j) = f3_32(:,j) -  shift_3_32(j);
        end

        % Create the shifting column and concatenate it
        f3_1 = [f3_1, abs(shift_3_1'.*ones(size(f3_1,1),size(f3_1,2)))];
        f3_2 = [f3_2, abs(shift_3_2'.*ones(size(f3_2,1),size(f3_2,2)))];
        f3_3 = [f3_3, abs(shift_3_3'.*ones(size(f3_3,1),size(f3_3,2)))];
        f3_4 = [f3_4, abs(shift_3_4'.*ones(size(f3_4,1),size(f3_4,2)))];
        f3_5 = [f3_5, abs(shift_3_5'.*ones(size(f3_5,1),size(f3_5,2)))];
        f3_6 = [f3_6, abs(shift_3_6'.*ones(size(f3_6,1),size(f3_6,2)))];
        f3_7 = [f3_7, abs(shift_3_7'.*ones(size(f3_7,1),size(f3_7,2)))];
        f3_8 = [f3_8, abs(shift_3_8'.*ones(size(f3_8,1),size(f3_8,2)))];
        f3_9 = [f3_9, abs(shift_3_9'.*ones(size(f3_9,1),size(f3_9,2)))];
        f3_10 = [f3_10, abs(shift_3_10'.*ones(size(f3_10,1),size(f3_10,2)))];
        f3_11 = [f3_11, abs(shift_3_11'.*ones(size(f3_11,1),size(f3_11,2)))];
        f3_12 = [f3_12, abs(shift_3_12'.*ones(size(f3_12,1),size(f3_12,2)))];
        f3_13 = [f3_13, abs(shift_3_13'.*ones(size(f3_13,1),size(f3_13,2)))];
        f3_14 = [f3_14, abs(shift_3_14'.*ones(size(f3_14,1),size(f3_14,2)))];
        f3_15 = [f3_15, abs(shift_3_15'.*ones(size(f3_15,1),size(f3_15,2)))];
        f3_16 = [f3_16, abs(shift_3_16'.*ones(size(f3_16,1),size(f3_16,2)))];
        f3_17 = [f3_17, abs(shift_3_17'.*ones(size(f3_17,1),size(f3_17,2)))];
        f3_18 = [f3_18, abs(shift_3_18'.*ones(size(f3_18,1),size(f3_18,2)))];
        f3_19 = [f3_19, abs(shift_3_19'.*ones(size(f3_19,1),size(f3_19,2)))];
        f3_20 = [f3_20, abs(shift_3_20'.*ones(size(f3_20,1),size(f3_20,2)))];
        f3_21 = [f3_21, abs(shift_3_21'.*ones(size(f3_21,1),size(f3_21,2)))];
        f3_22 = [f3_22, abs(shift_3_22'.*ones(size(f3_22,1),size(f3_22,2)))];
        f3_23 = [f3_23, abs(shift_3_23'.*ones(size(f3_23,1),size(f3_23,2)))];
        f3_24 = [f3_24, abs(shift_3_24'.*ones(size(f3_24,1),size(f3_24,2)))];
        f3_25 = [f3_25, abs(shift_3_25'.*ones(size(f3_25,1),size(f3_25,2)))];
        f3_26 = [f3_26, abs(shift_3_26'.*ones(size(f3_26,1),size(f3_26,2)))];
        f3_27 = [f3_27, abs(shift_3_27'.*ones(size(f3_27,1),size(f3_27,2)))];
        f3_28 = [f3_28, abs(shift_3_28'.*ones(size(f3_28,1),size(f3_28,2)))];
        f3_29 = [f3_29, abs(shift_3_29'.*ones(size(f3_29,1),size(f3_29,2)))];
        f3_30 = [f3_30, abs(shift_3_30'.*ones(size(f3_30,1),size(f3_30,2)))];
        f3_31 = [f3_31, abs(shift_3_31'.*ones(size(f3_31,1),size(f3_31,2)))];
        f3_32 = [f3_32, abs(shift_3_32'.*ones(size(f3_32,1),size(f3_32,2)))];

        out_1_3 = im_3_1*f3_1;
        out_2_3 = im_3_2*f3_2;
        out_3_3 = im_3_3*f3_3;
        out_4_3 = im_3_4*f3_4;
        out_5_3 = im_3_5*f3_5;
        out_6_3 = im_3_6*f3_6;
        out_7_3 = im_3_7*f3_7;
        out_8_3 = im_3_8*f3_8;
        out_9_3 = im_3_9*f3_9;
        out_10_3 = im_3_10*f3_10;
        out_11_3 = im_3_11*f3_11;
        out_12_3 = im_3_12*f3_12;
        out_13_3 = im_3_13*f3_13;
        out_14_3 = im_3_14*f3_14;
        out_15_3 = im_3_15*f3_15;
        out_16_3 = im_3_16*f3_16;
        out_17_3 = im_3_17*f3_17;
        out_18_3 = im_3_18*f3_18;
        out_19_3 = im_3_19*f3_19;
        out_20_3 = im_3_20*f3_20;
        out_21_3 = im_3_21*f3_21;
        out_22_3 = im_3_22*f3_22;
        out_23_3 = im_3_23*f3_23;
        out_24_3 = im_3_24*f3_24;
        out_25_3 = im_3_25*f3_25;
        out_26_3 = im_3_26*f3_26;
        out_27_3 = im_3_27*f3_27;
        out_28_3 = im_3_28*f3_28;
        out_29_3 = im_3_29*f3_29;
        out_30_3 = im_3_30*f3_30;
        out_31_3 = im_3_31*f3_31;
        out_32_3 = im_3_32*f3_32;

        output_third_layer_with_shifting = out_1_3 + out_2_3 + out_3_3 + out_4_3 +...
            out_5_3 + out_6_3 + out_7_3 + out_8_3 + out_9_3 + out_10_3 + out_11_3 +...
            out_12_3 + out_13_3 + out_14_3 + out_15_3 + out_16_3 + out_17_3 + out_18_3 +...
            out_19_3 + out_20_3 + out_21_3 + out_22_3 + out_23_3 + out_24_3 + out_25_3 +...
            out_26_3 + out_27_3 + out_28_3 + out_29_3 + out_30_3 + out_31_3 + out_32_3;


        shifting_matrix3 = output_third_layer_with_shifting(:,65:end);
        output_third_layer_with_shifting = output_third_layer_with_shifting(:,1:64);

        % Each col is an output filter (its flattening happened row-by-row)
        output_after_shifting3 = output_third_layer_with_shifting - shifting_matrix3;

        % Re-shape it to pass it through pooling and relu
        output_3_shifting = permute(reshape(output_after_shifting3,[8,8,64]),[2 1 3]);


        minValue = min([f3_1(:);f3_2(:);f3_3(:);f3_4(:); f3_5(:); f3_6(:); f3_7(:);...
            f3_8(:); f3_9(:); f3_10(:); f3_11(:); f3_12(:); f3_13(:); f3_14(:);...
            f3_15(:); f3_16(:); f3_17(:); f3_18(:); f3_19(:); f3_20(:); f3_21(:);...
            f3_22(:); f3_23(:); f3_24(:); f3_25(:); f3_26(:); f3_27(:); f3_28(:);...
            f3_29(:); f3_30(:); f3_31(:); f3_32(:)]);

        maxValue = max([f3_1(:);f3_2(:);f3_3(:);f3_4(:); f3_5(:); f3_6(:); f3_7(:);...
            f3_8(:); f3_9(:); f3_10(:); f3_11(:); f3_12(:); f3_13(:); f3_14(:);...
            f3_15(:); f3_16(:); f3_17(:); f3_18(:); f3_19(:); f3_20(:); f3_21(:);...
            f3_22(:); f3_23(:); f3_24(:); f3_25(:); f3_26(:); f3_27(:); f3_28(:);...
            f3_29(:); f3_30(:); f3_31(:); f3_32(:)]);

        a_h = (Gon - Goff)/(maxValue - minValue);
        b_h = Gon - a_h *(maxValue);
        G_f3_1 = a_h .* f3_1 + b_h;
        G_f3_2 = a_h .* f3_2 + b_h;
        G_f3_3 = a_h .* f3_3 + b_h;
        G_f3_4 = a_h .* f3_4 + b_h; 
        G_f3_5 = a_h .* f3_5 + b_h; 
        G_f3_6 = a_h .* f3_6 + b_h; 
        G_f3_7 = a_h .* f3_7 + b_h; 
        G_f3_8 = a_h .* f3_8 + b_h; 
        G_f3_9 = a_h .* f3_9 + b_h; 
        G_f3_10 = a_h .* f3_10 + b_h; 
        G_f3_11 = a_h .* f3_11 + b_h; 
        G_f3_12 = a_h .* f3_12 + b_h; 
        G_f3_13 = a_h .* f3_13 + b_h; 
        G_f3_14 = a_h .* f3_14 + b_h; 
        G_f3_15 = a_h .* f3_15 + b_h; 
        G_f3_16 = a_h .* f3_16 + b_h; 
        G_f3_17 = a_h .* f3_17 + b_h; 
        G_f3_18 = a_h .* f3_18 + b_h; 
        G_f3_19 = a_h .* f3_19 + b_h; 
        G_f3_20 = a_h .* f3_20 + b_h; 
        G_f3_21 = a_h .* f3_21 + b_h; 
        G_f3_22 = a_h .* f3_22 + b_h; 
        G_f3_23 = a_h .* f3_23 + b_h; 
        G_f3_24 = a_h .* f3_24 + b_h; 
        G_f3_25 = a_h .* f3_25 + b_h; 
        G_f3_26 = a_h .* f3_26 + b_h; 
        G_f3_27 = a_h .* f3_27 + b_h; 
        G_f3_28 = a_h .* f3_28 + b_h; 
        G_f3_29 = a_h .* f3_29 + b_h; 
        G_f3_30 = a_h .* f3_30 + b_h; 
        G_f3_31 = a_h .* f3_31 + b_h; 
        G_f3_32 = a_h .* f3_32 + b_h; 


            new_val_f3_1 = cell(1,1); 
            new_val_f3_2 = cell(1,1); 
            new_val_f3_3 = cell(1,1); 
            new_val_f3_4 = cell(1,1); 
            new_val_f3_5 = cell(1,1); 
            new_val_f3_6 = cell(1,1); 
            new_val_f3_7 = cell(1,1); 
            new_val_f3_8 = cell(1,1); 
            new_val_f3_9 = cell(1,1); 
            new_val_f3_10 = cell(1,1); 
            new_val_f3_11 = cell(1,1); 
            new_val_f3_12 = cell(1,1); 
            new_val_f3_13 = cell(1,1); 
            new_val_f3_14 = cell(1,1); 
            new_val_f3_15 = cell(1,1); 
            new_val_f3_16 = cell(1,1); 
            new_val_f3_17 = cell(1,1); 
            new_val_f3_18 = cell(1,1); 
            new_val_f3_19 = cell(1,1); 
            new_val_f3_20 = cell(1,1); 
            new_val_f3_21 = cell(1,1); 
            new_val_f3_22 = cell(1,1); 
            new_val_f3_23 = cell(1,1); 
            new_val_f3_24 = cell(1,1); 
            new_val_f3_25 = cell(1,1); 
            new_val_f3_26 = cell(1,1); 
            new_val_f3_27 = cell(1,1); 
            new_val_f3_28 = cell(1,1); 
            new_val_f3_29 = cell(1,1); 
            new_val_f3_30 = cell(1,1); 
            new_val_f3_31 = cell(1,1); 
            new_val_f3_32 = cell(1,1); 
            
            
            for i = 1: length(edges)-1
                range_mean = [edges(i) edges(i+1)];
                pd = makedist('normal','mu', mean(range_mean), 'sigma', mean(range_mean) + (variation*mean(range_mean)));
                upper = mean(range_mean) + (variation*mean(range_mean));
                lower = mean(range_mean) - (variation*mean(range_mean));
                t = truncate(pd, lower,upper);

                
                new_val_f3_1{i} = random(t,1,1);  
                new_val_f3_2{i} = random(t,1,1); 
                new_val_f3_3{i} = random(t,1,1); 
                new_val_f3_4{i} = random(t,1,1); 
                new_val_f3_5{i} = random(t,1,1); 
                new_val_f3_6{i} = random(t,1,1); 
                new_val_f3_7{i} = random(t,1,1); 
                new_val_f3_8{i} = random(t,1,1); 
                new_val_f3_9{i} = random(t,1,1); 
                new_val_f3_10{i} = random(t,1,1); 
                new_val_f3_11{i} = random(t,1,1); 
                new_val_f3_12{i} = random(t,1,1); 
                new_val_f3_13{i} = random(t,1,1); 
                new_val_f3_14{i} = random(t,1,1); 
                new_val_f3_15{i} = random(t,1,1); 
                new_val_f3_16{i} = random(t,1,1); 
                new_val_f3_17{i} = random(t,1,1); 
                new_val_f3_18{i} = random(t,1,1); 
                new_val_f3_19{i} = random(t,1,1); 
                new_val_f3_20{i} = random(t,1,1); 
                new_val_f3_21{i} = random(t,1,1); 
                new_val_f3_22{i} = random(t,1,1); 
                new_val_f3_23{i} = random(t,1,1); 
                new_val_f3_24{i} = random(t,1,1); 
                new_val_f3_25{i} = random(t,1,1); 
                new_val_f3_26{i} = random(t,1,1); 
                new_val_f3_27{i} = random(t,1,1); 
                new_val_f3_28{i} = random(t,1,1); 
                new_val_f3_29{i} = random(t,1,1); 
                new_val_f3_30{i} = random(t,1,1); 
                new_val_f3_31{i} = random(t,1,1); 
                new_val_f3_32{i} = random(t,1,1); 

%                 new_val_f3_1{i} = mean(range_mean); 
%                 new_val_f3_2{i} = mean(range_mean); 
%                 new_val_f3_3{i} = mean(range_mean); 
%                 new_val_f3_4{i} = mean(range_mean); 
%                 new_val_f3_5{i} = mean(range_mean); 
%                 new_val_f3_6{i} = mean(range_mean); 
%                 new_val_f3_7{i} = mean(range_mean); 
%                 new_val_f3_8{i} = mean(range_mean); 
%                 new_val_f3_9{i} = mean(range_mean); 
%                 new_val_f3_10{i} = mean(range_mean); 
%                 new_val_f3_11{i} = mean(range_mean); 
%                 new_val_f3_12{i} = mean(range_mean); 
%                 new_val_f3_13{i} = mean(range_mean); 
%                 new_val_f3_14{i} = mean(range_mean); 
%                 new_val_f3_15{i} = mean(range_mean); 
%                 new_val_f3_16{i} = mean(range_mean); 
%                 new_val_f3_17{i} = mean(range_mean); 
%                 new_val_f3_18{i} = mean(range_mean); 
%                 new_val_f3_19{i} = mean(range_mean); 
%                 new_val_f3_20{i} = mean(range_mean); 
%                 new_val_f3_21{i} = mean(range_mean); 
%                 new_val_f3_22{i} = mean(range_mean); 
%                 new_val_f3_23{i} = mean(range_mean); 
%                 new_val_f3_24{i} = mean(range_mean); 
%                 new_val_f3_25{i} = mean(range_mean); 
%                 new_val_f3_26{i} = mean(range_mean); 
%                 new_val_f3_27{i} = mean(range_mean); 
%                 new_val_f3_28{i} = mean(range_mean); 
%                 new_val_f3_29{i} = mean(range_mean); 
%                 new_val_f3_30{i} = mean(range_mean); 
%                 new_val_f3_31{i} = mean(range_mean); 
%                 new_val_f3_32{i} = mean(range_mean); 

            end
        
            new_val_f3_1 = cell2mat(new_val_f3_1); 
            new_val_f3_2 = cell2mat(new_val_f3_2); 
            new_val_f3_3 = cell2mat(new_val_f3_3); 
            new_val_f3_4 = cell2mat(new_val_f3_4); 
            new_val_f3_5 = cell2mat(new_val_f3_5); 
            new_val_f3_6 = cell2mat(new_val_f3_6); 
            new_val_f3_7 = cell2mat(new_val_f3_7); 
            new_val_f3_8 = cell2mat(new_val_f3_8); 
            new_val_f3_9 = cell2mat(new_val_f3_9); 
            new_val_f3_10 = cell2mat(new_val_f3_10); 
            new_val_f3_11 = cell2mat(new_val_f3_11); 
            new_val_f3_12 = cell2mat(new_val_f3_12); 
            new_val_f3_13 = cell2mat(new_val_f3_13); 
            new_val_f3_14 = cell2mat(new_val_f3_14); 
            new_val_f3_15 = cell2mat(new_val_f3_15); 
            new_val_f3_16 = cell2mat(new_val_f3_16); 
            new_val_f3_17 = cell2mat(new_val_f3_17); 
            new_val_f3_18 = cell2mat(new_val_f3_18); 
            new_val_f3_19 = cell2mat(new_val_f3_19); 
            new_val_f3_20 = cell2mat(new_val_f3_20); 
            new_val_f3_21 = cell2mat(new_val_f3_21); 
            new_val_f3_22 = cell2mat(new_val_f3_22); 
            new_val_f3_23 = cell2mat(new_val_f3_23); 
            new_val_f3_24 = cell2mat(new_val_f3_24); 
            new_val_f3_25 = cell2mat(new_val_f3_25); 
            new_val_f3_26 = cell2mat(new_val_f3_26); 
            new_val_f3_27 = cell2mat(new_val_f3_27); 
            new_val_f3_28 = cell2mat(new_val_f3_28); 
            new_val_f3_29 = cell2mat(new_val_f3_29); 
            new_val_f3_30 = cell2mat(new_val_f3_30); 
            new_val_f3_31 = cell2mat(new_val_f3_31); 
            new_val_f3_32 = cell2mat(new_val_f3_32); 
        

            
            G_f3_1_q = zeros(size(G_f3_1,1),size(G_f3_1,2)); 
            G_f3_2_q = zeros(size(G_f3_2,1),size(G_f3_2,2)); 
            G_f3_3_q = zeros(size(G_f3_3,1),size(G_f3_3,2)); 
            G_f3_4_q = zeros(size(G_f3_4,1),size(G_f3_4,2)); 
            G_f3_5_q = zeros(size(G_f3_5,1),size(G_f3_5,2)); 
            G_f3_6_q = zeros(size(G_f3_6,1),size(G_f3_6,2)); 
            G_f3_7_q = zeros(size(G_f3_7,1),size(G_f3_7,2)); 
            G_f3_8_q = zeros(size(G_f3_8,1),size(G_f3_8,2)); 
            G_f3_9_q = zeros(size(G_f3_9,1),size(G_f3_9,2)); 
            G_f3_10_q = zeros(size(G_f3_10,1),size(G_f3_10,2)); 
            G_f3_11_q = zeros(size(G_f3_11,1),size(G_f3_11,2)); 
            G_f3_12_q = zeros(size(G_f3_12,1),size(G_f3_12,2)); 
            G_f3_13_q = zeros(size(G_f3_13,1),size(G_f3_13,2)); 
            G_f3_14_q = zeros(size(G_f3_14,1),size(G_f3_14,2)); 
            G_f3_15_q = zeros(size(G_f3_15,1),size(G_f3_15,2)); 
            G_f3_16_q = zeros(size(G_f3_16,1),size(G_f3_16,2)); 
            G_f3_17_q = zeros(size(G_f3_17,1),size(G_f3_17,2)); 
            G_f3_18_q = zeros(size(G_f3_18,1),size(G_f3_18,2)); 
            G_f3_19_q = zeros(size(G_f3_19,1),size(G_f3_19,2)); 
            G_f3_20_q = zeros(size(G_f3_20,1),size(G_f3_20,2)); 
            G_f3_21_q = zeros(size(G_f3_21,1),size(G_f3_21,2)); 
            G_f3_22_q = zeros(size(G_f3_22,1),size(G_f3_22,2)); 
            G_f3_23_q = zeros(size(G_f3_23,1),size(G_f3_23,2)); 
            G_f3_24_q = zeros(size(G_f3_24,1),size(G_f3_24,2)); 
            G_f3_25_q = zeros(size(G_f3_25,1),size(G_f3_25,2)); 
            G_f3_26_q = zeros(size(G_f3_26,1),size(G_f3_26,2)); 
            G_f3_27_q = zeros(size(G_f3_27,1),size(G_f3_27,2)); 
            G_f3_28_q = zeros(size(G_f3_28,1),size(G_f3_28,2)); 
            G_f3_29_q = zeros(size(G_f3_29,1),size(G_f3_29,2)); 
            G_f3_30_q = zeros(size(G_f3_30,1),size(G_f3_30,2)); 
            G_f3_31_q = zeros(size(G_f3_31,1),size(G_f3_31,2)); 
            G_f3_32_q = zeros(size(G_f3_32,1),size(G_f3_32,2)); 
            
            
            
             for i = 1:size(G_f3_1,1) 
                for j = 1:size(G_f3_1,2) 
                    if G_f3_1(i,j) < edges(1)  
                        G_f3_1_q(i,j) = new_val_f3_1(1);  
                    elseif G_f3_1(i,j) >= edges(1) && G_f3_1(i,j) < edges(2)  
                        G_f3_1_q(i,j) = new_val_f3_1(1);  
                    elseif G_f3_1(i,j) >= edges(2) && G_f3_1(i,j) < edges(3)  
                        G_f3_1_q(i,j) = new_val_f3_1(2);    
                    elseif G_f3_1(i,j) >= edges(3) && G_f3_1(i,j) < edges(4)  
                        G_f3_1_q(i,j) = new_val_f3_1(3);  
                    elseif G_f3_1(i,j) >= edges(4) && G_f3_1(i,j) < edges(5)  
                        G_f3_1_q(i,j) = new_val_f3_1(4);   
                    elseif G_f3_1(i,j) >= edges(5) && G_f3_1(i,j) < edges(6)  
                        G_f3_1_q(i,j) = new_val_f3_1(5);     
                    elseif G_f3_1(i,j) >= edges(6) && G_f3_1(i,j) < edges(7)  
                        G_f3_1_q(i,j) = new_val_f3_1(6);  
                    elseif G_f3_1(i,j) >= edges(7) && G_f3_1(i,j) < edges(8)  
                        G_f3_1_q(i,j) = new_val_f3_1(7);   
                    elseif G_f3_1(i,j) >= edges(8) && G_f3_1(i,j) < edges(9)  
                        G_f3_1_q(i,j) = new_val_f3_1(8);     
                    elseif G_f3_1(i,j) >= edges(9) && G_f3_1(i,j) < edges(10)  
                        G_f3_1_q(i,j) = new_val_f3_1(9);  
                    elseif G_f3_1(i,j) >= edges(10) && G_f3_1(i,j) < edges(11)  
                        G_f3_1_q(i,j) = new_val_f3_1(10);  
                    elseif G_f3_1(i,j) >= edges(11) && G_f3_1(i,j) < edges(12)  
                        G_f3_1_q(i,j) = new_val_f3_1(11);  
                    elseif G_f3_1(i,j) >= edges(12) && G_f3_1(i,j) < edges(13)  
                        G_f3_1_q(i,j) = new_val_f3_1(12);  
                    elseif G_f3_1(i,j) >= edges(13) && G_f3_1(i,j) < edges(14)   
                         G_f3_1_q(i,j) = new_val_f3_1(13);   
                    elseif G_f3_1(i,j) >= edges(14) && G_f3_1(i,j) < edges(15)  
                        G_f3_1_q(i,j) = new_val_f3_1(14);  
                    elseif G_f3_1(i,j) >= edges(15) && G_f3_1(i,j) < edges(16)  
                        G_f3_1_q(i,j) = new_val_f3_1(15);  
                    elseif G_f3_1(i,j) >= edges(16) && G_f3_1(i,j) <= edges(17)   
                         G_f3_1_q(i,j) = new_val_f3_1(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_2,1) 
                for j = 1:size(G_f3_2,2) 
                    if G_f3_2(i,j) < edges(1)  
                        G_f3_2_q(i,j) = new_val_f3_2(1);  
                    elseif G_f3_2(i,j) >= edges(1) && G_f3_2(i,j) < edges(2)  
                        G_f3_2_q(i,j) = new_val_f3_2(1);  
                    elseif G_f3_2(i,j) >= edges(2) && G_f3_2(i,j) < edges(3)  
                        G_f3_2_q(i,j) = new_val_f3_2(2);    
                    elseif G_f3_2(i,j) >= edges(3) && G_f3_2(i,j) < edges(4)  
                        G_f3_2_q(i,j) = new_val_f3_2(3);  
                    elseif G_f3_2(i,j) >= edges(4) && G_f3_2(i,j) < edges(5)  
                        G_f3_2_q(i,j) = new_val_f3_2(4);   
                    elseif G_f3_2(i,j) >= edges(5) && G_f3_2(i,j) < edges(6)  
                        G_f3_2_q(i,j) = new_val_f3_2(5);     
                    elseif G_f3_2(i,j) >= edges(6) && G_f3_2(i,j) < edges(7)  
                        G_f3_2_q(i,j) = new_val_f3_2(6);  
                    elseif G_f3_2(i,j) >= edges(7) && G_f3_2(i,j) < edges(8)  
                        G_f3_2_q(i,j) = new_val_f3_2(7);   
                    elseif G_f3_2(i,j) >= edges(8) && G_f3_2(i,j) < edges(9)  
                        G_f3_2_q(i,j) = new_val_f3_2(8);     
                    elseif G_f3_2(i,j) >= edges(9) && G_f3_2(i,j) < edges(10)  
                        G_f3_2_q(i,j) = new_val_f3_2(9);  
                    elseif G_f3_2(i,j) >= edges(10) && G_f3_2(i,j) < edges(11)  
                        G_f3_2_q(i,j) = new_val_f3_2(10);  
                    elseif G_f3_2(i,j) >= edges(11) && G_f3_2(i,j) < edges(12)  
                        G_f3_2_q(i,j) = new_val_f3_2(11);  
                    elseif G_f3_2(i,j) >= edges(12) && G_f3_2(i,j) < edges(13)  
                        G_f3_2_q(i,j) = new_val_f3_2(12);  
                    elseif G_f3_2(i,j) >= edges(13) && G_f3_2(i,j) < edges(14)   
                         G_f3_2_q(i,j) = new_val_f3_2(13);   
                    elseif G_f3_2(i,j) >= edges(14) && G_f3_2(i,j) < edges(15)  
                        G_f3_2_q(i,j) = new_val_f3_2(14);  
                    elseif G_f3_2(i,j) >= edges(15) && G_f3_2(i,j) < edges(16)  
                        G_f3_2_q(i,j) = new_val_f3_2(15);  
                    elseif G_f3_2(i,j) >= edges(16) && G_f3_2(i,j) <= edges(17)   
                         G_f3_2_q(i,j) = new_val_f3_2(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_3,1) 
                for j = 1:size(G_f3_3,2) 
                    if G_f3_3(i,j) < edges(1)  
                        G_f3_3_q(i,j) = new_val_f3_3(1);  
                    elseif G_f3_3(i,j) >= edges(1) && G_f3_3(i,j) < edges(2)  
                        G_f3_3_q(i,j) = new_val_f3_3(1);  
                    elseif G_f3_3(i,j) >= edges(2) && G_f3_3(i,j) < edges(3)  
                        G_f3_3_q(i,j) = new_val_f3_3(2);    
                    elseif G_f3_3(i,j) >= edges(3) && G_f3_3(i,j) < edges(4)  
                        G_f3_3_q(i,j) = new_val_f3_3(3);  
                    elseif G_f3_3(i,j) >= edges(4) && G_f3_3(i,j) < edges(5)  
                        G_f3_3_q(i,j) = new_val_f3_3(4);   
                    elseif G_f3_3(i,j) >= edges(5) && G_f3_3(i,j) < edges(6)  
                        G_f3_3_q(i,j) = new_val_f3_3(5);     
                    elseif G_f3_3(i,j) >= edges(6) && G_f3_3(i,j) < edges(7)  
                        G_f3_3_q(i,j) = new_val_f3_3(6);  
                    elseif G_f3_3(i,j) >= edges(7) && G_f3_3(i,j) < edges(8)  
                        G_f3_3_q(i,j) = new_val_f3_3(7);   
                    elseif G_f3_3(i,j) >= edges(8) && G_f3_3(i,j) < edges(9)  
                        G_f3_3_q(i,j) = new_val_f3_3(8);     
                    elseif G_f3_3(i,j) >= edges(9) && G_f3_3(i,j) < edges(10)  
                        G_f3_3_q(i,j) = new_val_f3_3(9);  
                    elseif G_f3_3(i,j) >= edges(10) && G_f3_3(i,j) < edges(11)  
                        G_f3_3_q(i,j) = new_val_f3_3(10);  
                    elseif G_f3_3(i,j) >= edges(11) && G_f3_3(i,j) < edges(12)  
                        G_f3_3_q(i,j) = new_val_f3_3(11);  
                    elseif G_f3_3(i,j) >= edges(12) && G_f3_3(i,j) < edges(13)  
                        G_f3_3_q(i,j) = new_val_f3_3(12);  
                    elseif G_f3_3(i,j) >= edges(13) && G_f3_3(i,j) < edges(14)   
                         G_f3_3_q(i,j) = new_val_f3_3(13);   
                    elseif G_f3_3(i,j) >= edges(14) && G_f3_3(i,j) < edges(15)  
                        G_f3_3_q(i,j) = new_val_f3_3(14);  
                    elseif G_f3_3(i,j) >= edges(15) && G_f3_3(i,j) < edges(16)  
                        G_f3_3_q(i,j) = new_val_f3_3(15);  
                    elseif G_f3_3(i,j) >= edges(16) && G_f3_3(i,j) <= edges(17)   
                         G_f3_3_q(i,j) = new_val_f3_3(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_4,1) 
                for j = 1:size(G_f3_4,2) 
                    if G_f3_4(i,j) < edges(1)  
                        G_f3_4_q(i,j) = new_val_f3_4(1);  
                    elseif G_f3_4(i,j) >= edges(1) && G_f3_4(i,j) < edges(2)  
                        G_f3_4_q(i,j) = new_val_f3_4(1);  
                    elseif G_f3_4(i,j) >= edges(2) && G_f3_4(i,j) < edges(3)  
                        G_f3_4_q(i,j) = new_val_f3_4(2);    
                    elseif G_f3_4(i,j) >= edges(3) && G_f3_4(i,j) < edges(4)  
                        G_f3_4_q(i,j) = new_val_f3_4(3);  
                    elseif G_f3_4(i,j) >= edges(4) && G_f3_4(i,j) < edges(5)  
                        G_f3_4_q(i,j) = new_val_f3_4(4);   
                    elseif G_f3_4(i,j) >= edges(5) && G_f3_4(i,j) < edges(6)  
                        G_f3_4_q(i,j) = new_val_f3_4(5);     
                    elseif G_f3_4(i,j) >= edges(6) && G_f3_4(i,j) < edges(7)  
                        G_f3_4_q(i,j) = new_val_f3_4(6);  
                    elseif G_f3_4(i,j) >= edges(7) && G_f3_4(i,j) < edges(8)  
                        G_f3_4_q(i,j) = new_val_f3_4(7);   
                    elseif G_f3_4(i,j) >= edges(8) && G_f3_4(i,j) < edges(9)  
                        G_f3_4_q(i,j) = new_val_f3_4(8);     
                    elseif G_f3_4(i,j) >= edges(9) && G_f3_4(i,j) < edges(10)  
                        G_f3_4_q(i,j) = new_val_f3_4(9);  
                    elseif G_f3_4(i,j) >= edges(10) && G_f3_4(i,j) < edges(11)  
                        G_f3_4_q(i,j) = new_val_f3_4(10);  
                    elseif G_f3_4(i,j) >= edges(11) && G_f3_4(i,j) < edges(12)  
                        G_f3_4_q(i,j) = new_val_f3_4(11);  
                    elseif G_f3_4(i,j) >= edges(12) && G_f3_4(i,j) < edges(13)  
                        G_f3_4_q(i,j) = new_val_f3_4(12);  
                    elseif G_f3_4(i,j) >= edges(13) && G_f3_4(i,j) < edges(14)   
                         G_f3_4_q(i,j) = new_val_f3_4(13);   
                    elseif G_f3_4(i,j) >= edges(14) && G_f3_4(i,j) < edges(15)  
                        G_f3_4_q(i,j) = new_val_f3_4(14);  
                    elseif G_f3_4(i,j) >= edges(15) && G_f3_4(i,j) < edges(16)  
                        G_f3_4_q(i,j) = new_val_f3_4(15);  
                    elseif G_f3_4(i,j) >= edges(16) && G_f3_4(i,j) <= edges(17)   
                         G_f3_4_q(i,j) = new_val_f3_4(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_5,1) 
                for j = 1:size(G_f3_5,2) 
                    if G_f3_5(i,j) < edges(1)  
                        G_f3_5_q(i,j) = new_val_f3_5(1);  
                    elseif G_f3_5(i,j) >= edges(1) && G_f3_5(i,j) < edges(2)  
                        G_f3_5_q(i,j) = new_val_f3_5(1);  
                    elseif G_f3_5(i,j) >= edges(2) && G_f3_5(i,j) < edges(3)  
                        G_f3_5_q(i,j) = new_val_f3_5(2);    
                    elseif G_f3_5(i,j) >= edges(3) && G_f3_5(i,j) < edges(4)  
                        G_f3_5_q(i,j) = new_val_f3_5(3);  
                    elseif G_f3_5(i,j) >= edges(4) && G_f3_5(i,j) < edges(5)  
                        G_f3_5_q(i,j) = new_val_f3_5(4);   
                    elseif G_f3_5(i,j) >= edges(5) && G_f3_5(i,j) < edges(6)  
                        G_f3_5_q(i,j) = new_val_f3_5(5);     
                    elseif G_f3_5(i,j) >= edges(6) && G_f3_5(i,j) < edges(7)  
                        G_f3_5_q(i,j) = new_val_f3_5(6);  
                    elseif G_f3_5(i,j) >= edges(7) && G_f3_5(i,j) < edges(8)  
                        G_f3_5_q(i,j) = new_val_f3_5(7);   
                    elseif G_f3_5(i,j) >= edges(8) && G_f3_5(i,j) < edges(9)  
                        G_f3_5_q(i,j) = new_val_f3_5(8);     
                    elseif G_f3_5(i,j) >= edges(9) && G_f3_5(i,j) < edges(10)  
                        G_f3_5_q(i,j) = new_val_f3_5(9);  
                    elseif G_f3_5(i,j) >= edges(10) && G_f3_5(i,j) < edges(11)  
                        G_f3_5_q(i,j) = new_val_f3_5(10);  
                    elseif G_f3_5(i,j) >= edges(11) && G_f3_5(i,j) < edges(12)  
                        G_f3_5_q(i,j) = new_val_f3_5(11);  
                    elseif G_f3_5(i,j) >= edges(12) && G_f3_5(i,j) < edges(13)  
                        G_f3_5_q(i,j) = new_val_f3_5(12);  
                    elseif G_f3_5(i,j) >= edges(13) && G_f3_5(i,j) < edges(14)   
                         G_f3_5_q(i,j) = new_val_f3_5(13);   
                    elseif G_f3_5(i,j) >= edges(14) && G_f3_5(i,j) < edges(15)  
                        G_f3_5_q(i,j) = new_val_f3_5(14);  
                    elseif G_f3_5(i,j) >= edges(15) && G_f3_5(i,j) < edges(16)  
                        G_f3_5_q(i,j) = new_val_f3_5(15);  
                    elseif G_f3_5(i,j) >= edges(16) && G_f3_5(i,j) <= edges(17)   
                         G_f3_5_q(i,j) = new_val_f3_5(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_6,1) 
                for j = 1:size(G_f3_6,2) 
                    if G_f3_6(i,j) < edges(1)  
                        G_f3_6_q(i,j) = new_val_f3_6(1);  
                    elseif G_f3_6(i,j) >= edges(1) && G_f3_6(i,j) < edges(2)  
                        G_f3_6_q(i,j) = new_val_f3_6(1);  
                    elseif G_f3_6(i,j) >= edges(2) && G_f3_6(i,j) < edges(3)  
                        G_f3_6_q(i,j) = new_val_f3_6(2);    
                    elseif G_f3_6(i,j) >= edges(3) && G_f3_6(i,j) < edges(4)  
                        G_f3_6_q(i,j) = new_val_f3_6(3);  
                    elseif G_f3_6(i,j) >= edges(4) && G_f3_6(i,j) < edges(5)  
                        G_f3_6_q(i,j) = new_val_f3_6(4);   
                    elseif G_f3_6(i,j) >= edges(5) && G_f3_6(i,j) < edges(6)  
                        G_f3_6_q(i,j) = new_val_f3_6(5);     
                    elseif G_f3_6(i,j) >= edges(6) && G_f3_6(i,j) < edges(7)  
                        G_f3_6_q(i,j) = new_val_f3_6(6);  
                    elseif G_f3_6(i,j) >= edges(7) && G_f3_6(i,j) < edges(8)  
                        G_f3_6_q(i,j) = new_val_f3_6(7);   
                    elseif G_f3_6(i,j) >= edges(8) && G_f3_6(i,j) < edges(9)  
                        G_f3_6_q(i,j) = new_val_f3_6(8);     
                    elseif G_f3_6(i,j) >= edges(9) && G_f3_6(i,j) < edges(10)  
                        G_f3_6_q(i,j) = new_val_f3_6(9);  
                    elseif G_f3_6(i,j) >= edges(10) && G_f3_6(i,j) < edges(11)  
                        G_f3_6_q(i,j) = new_val_f3_6(10);  
                    elseif G_f3_6(i,j) >= edges(11) && G_f3_6(i,j) < edges(12)  
                        G_f3_6_q(i,j) = new_val_f3_6(11);  
                    elseif G_f3_6(i,j) >= edges(12) && G_f3_6(i,j) < edges(13)  
                        G_f3_6_q(i,j) = new_val_f3_6(12);  
                    elseif G_f3_6(i,j) >= edges(13) && G_f3_6(i,j) < edges(14)   
                         G_f3_6_q(i,j) = new_val_f3_6(13);   
                    elseif G_f3_6(i,j) >= edges(14) && G_f3_6(i,j) < edges(15)  
                        G_f3_6_q(i,j) = new_val_f3_6(14);  
                    elseif G_f3_6(i,j) >= edges(15) && G_f3_6(i,j) < edges(16)  
                        G_f3_6_q(i,j) = new_val_f3_6(15);  
                    elseif G_f3_6(i,j) >= edges(16) && G_f3_6(i,j) <= edges(17)   
                         G_f3_6_q(i,j) = new_val_f3_6(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_7,1) 
                for j = 1:size(G_f3_7,2) 
                    if G_f3_7(i,j) < edges(1)  
                        G_f3_7_q(i,j) = new_val_f3_7(1);  
                    elseif G_f3_7(i,j) >= edges(1) && G_f3_7(i,j) < edges(2)  
                        G_f3_7_q(i,j) = new_val_f3_7(1);  
                    elseif G_f3_7(i,j) >= edges(2) && G_f3_7(i,j) < edges(3)  
                        G_f3_7_q(i,j) = new_val_f3_7(2);    
                    elseif G_f3_7(i,j) >= edges(3) && G_f3_7(i,j) < edges(4)  
                        G_f3_7_q(i,j) = new_val_f3_7(3);  
                    elseif G_f3_7(i,j) >= edges(4) && G_f3_7(i,j) < edges(5)  
                        G_f3_7_q(i,j) = new_val_f3_7(4);   
                    elseif G_f3_7(i,j) >= edges(5) && G_f3_7(i,j) < edges(6)  
                        G_f3_7_q(i,j) = new_val_f3_7(5);     
                    elseif G_f3_7(i,j) >= edges(6) && G_f3_7(i,j) < edges(7)  
                        G_f3_7_q(i,j) = new_val_f3_7(6);  
                    elseif G_f3_7(i,j) >= edges(7) && G_f3_7(i,j) < edges(8)  
                        G_f3_7_q(i,j) = new_val_f3_7(7);   
                    elseif G_f3_7(i,j) >= edges(8) && G_f3_7(i,j) < edges(9)  
                        G_f3_7_q(i,j) = new_val_f3_7(8);     
                    elseif G_f3_7(i,j) >= edges(9) && G_f3_7(i,j) < edges(10)  
                        G_f3_7_q(i,j) = new_val_f3_7(9);  
                    elseif G_f3_7(i,j) >= edges(10) && G_f3_7(i,j) < edges(11)  
                        G_f3_7_q(i,j) = new_val_f3_7(10);  
                    elseif G_f3_7(i,j) >= edges(11) && G_f3_7(i,j) < edges(12)  
                        G_f3_7_q(i,j) = new_val_f3_7(11);  
                    elseif G_f3_7(i,j) >= edges(12) && G_f3_7(i,j) < edges(13)  
                        G_f3_7_q(i,j) = new_val_f3_7(12);  
                    elseif G_f3_7(i,j) >= edges(13) && G_f3_7(i,j) < edges(14)   
                         G_f3_7_q(i,j) = new_val_f3_7(13);   
                    elseif G_f3_7(i,j) >= edges(14) && G_f3_7(i,j) < edges(15)  
                        G_f3_7_q(i,j) = new_val_f3_7(14);  
                    elseif G_f3_7(i,j) >= edges(15) && G_f3_7(i,j) < edges(16)  
                        G_f3_7_q(i,j) = new_val_f3_7(15);  
                    elseif G_f3_7(i,j) >= edges(16) && G_f3_7(i,j) <= edges(17)   
                         G_f3_7_q(i,j) = new_val_f3_7(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_8,1) 
                for j = 1:size(G_f3_8,2) 
                    if G_f3_8(i,j) < edges(1)  
                        G_f3_8_q(i,j) = new_val_f3_8(1);  
                    elseif G_f3_8(i,j) >= edges(1) && G_f3_8(i,j) < edges(2)  
                        G_f3_8_q(i,j) = new_val_f3_8(1);  
                    elseif G_f3_8(i,j) >= edges(2) && G_f3_8(i,j) < edges(3)  
                        G_f3_8_q(i,j) = new_val_f3_8(2);    
                    elseif G_f3_8(i,j) >= edges(3) && G_f3_8(i,j) < edges(4)  
                        G_f3_8_q(i,j) = new_val_f3_8(3);  
                    elseif G_f3_8(i,j) >= edges(4) && G_f3_8(i,j) < edges(5)  
                        G_f3_8_q(i,j) = new_val_f3_8(4);   
                    elseif G_f3_8(i,j) >= edges(5) && G_f3_8(i,j) < edges(6)  
                        G_f3_8_q(i,j) = new_val_f3_8(5);     
                    elseif G_f3_8(i,j) >= edges(6) && G_f3_8(i,j) < edges(7)  
                        G_f3_8_q(i,j) = new_val_f3_8(6);  
                    elseif G_f3_8(i,j) >= edges(7) && G_f3_8(i,j) < edges(8)  
                        G_f3_8_q(i,j) = new_val_f3_8(7);   
                    elseif G_f3_8(i,j) >= edges(8) && G_f3_8(i,j) < edges(9)  
                        G_f3_8_q(i,j) = new_val_f3_8(8);     
                    elseif G_f3_8(i,j) >= edges(9) && G_f3_8(i,j) < edges(10)  
                        G_f3_8_q(i,j) = new_val_f3_8(9);  
                    elseif G_f3_8(i,j) >= edges(10) && G_f3_8(i,j) < edges(11)  
                        G_f3_8_q(i,j) = new_val_f3_8(10);  
                    elseif G_f3_8(i,j) >= edges(11) && G_f3_8(i,j) < edges(12)  
                        G_f3_8_q(i,j) = new_val_f3_8(11);  
                    elseif G_f3_8(i,j) >= edges(12) && G_f3_8(i,j) < edges(13)  
                        G_f3_8_q(i,j) = new_val_f3_8(12);  
                    elseif G_f3_8(i,j) >= edges(13) && G_f3_8(i,j) < edges(14)   
                         G_f3_8_q(i,j) = new_val_f3_8(13);   
                    elseif G_f3_8(i,j) >= edges(14) && G_f3_8(i,j) < edges(15)  
                        G_f3_8_q(i,j) = new_val_f3_8(14);  
                    elseif G_f3_8(i,j) >= edges(15) && G_f3_8(i,j) < edges(16)  
                        G_f3_8_q(i,j) = new_val_f3_8(15);  
                    elseif G_f3_8(i,j) >= edges(16) && G_f3_8(i,j) <= edges(17)   
                         G_f3_8_q(i,j) = new_val_f3_8(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_9,1) 
                for j = 1:size(G_f3_9,2) 
                    if G_f3_9(i,j) < edges(1)  
                        G_f3_9_q(i,j) = new_val_f3_9(1);  
                    elseif G_f3_9(i,j) >= edges(1) && G_f3_9(i,j) < edges(2)  
                        G_f3_9_q(i,j) = new_val_f3_9(1);  
                    elseif G_f3_9(i,j) >= edges(2) && G_f3_9(i,j) < edges(3)  
                        G_f3_9_q(i,j) = new_val_f3_9(2);    
                    elseif G_f3_9(i,j) >= edges(3) && G_f3_9(i,j) < edges(4)  
                        G_f3_9_q(i,j) = new_val_f3_9(3);  
                    elseif G_f3_9(i,j) >= edges(4) && G_f3_9(i,j) < edges(5)  
                        G_f3_9_q(i,j) = new_val_f3_9(4);   
                    elseif G_f3_9(i,j) >= edges(5) && G_f3_9(i,j) < edges(6)  
                        G_f3_9_q(i,j) = new_val_f3_9(5);     
                    elseif G_f3_9(i,j) >= edges(6) && G_f3_9(i,j) < edges(7)  
                        G_f3_9_q(i,j) = new_val_f3_9(6);  
                    elseif G_f3_9(i,j) >= edges(7) && G_f3_9(i,j) < edges(8)  
                        G_f3_9_q(i,j) = new_val_f3_9(7);   
                    elseif G_f3_9(i,j) >= edges(8) && G_f3_9(i,j) < edges(9)  
                        G_f3_9_q(i,j) = new_val_f3_9(8);     
                    elseif G_f3_9(i,j) >= edges(9) && G_f3_9(i,j) < edges(10)  
                        G_f3_9_q(i,j) = new_val_f3_9(9);  
                    elseif G_f3_9(i,j) >= edges(10) && G_f3_9(i,j) < edges(11)  
                        G_f3_9_q(i,j) = new_val_f3_9(10);  
                    elseif G_f3_9(i,j) >= edges(11) && G_f3_9(i,j) < edges(12)  
                        G_f3_9_q(i,j) = new_val_f3_9(11);  
                    elseif G_f3_9(i,j) >= edges(12) && G_f3_9(i,j) < edges(13)  
                        G_f3_9_q(i,j) = new_val_f3_9(12);  
                    elseif G_f3_9(i,j) >= edges(13) && G_f3_9(i,j) < edges(14)   
                         G_f3_9_q(i,j) = new_val_f3_9(13);   
                    elseif G_f3_9(i,j) >= edges(14) && G_f3_9(i,j) < edges(15)  
                        G_f3_9_q(i,j) = new_val_f3_9(14);  
                    elseif G_f3_9(i,j) >= edges(15) && G_f3_9(i,j) < edges(16)  
                        G_f3_9_q(i,j) = new_val_f3_9(15);  
                    elseif G_f3_9(i,j) >= edges(16) && G_f3_9(i,j) <= edges(17)   
                         G_f3_9_q(i,j) = new_val_f3_9(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_10,1) 
                for j = 1:size(G_f3_10,2) 
                    if G_f3_10(i,j) < edges(1)  
                        G_f3_10_q(i,j) = new_val_f3_10(1);  
                    elseif G_f3_10(i,j) >= edges(1) && G_f3_10(i,j) < edges(2)  
                        G_f3_10_q(i,j) = new_val_f3_10(1);  
                    elseif G_f3_10(i,j) >= edges(2) && G_f3_10(i,j) < edges(3)  
                        G_f3_10_q(i,j) = new_val_f3_10(2);    
                    elseif G_f3_10(i,j) >= edges(3) && G_f3_10(i,j) < edges(4)  
                        G_f3_10_q(i,j) = new_val_f3_10(3);  
                    elseif G_f3_10(i,j) >= edges(4) && G_f3_10(i,j) < edges(5)  
                        G_f3_10_q(i,j) = new_val_f3_10(4);   
                    elseif G_f3_10(i,j) >= edges(5) && G_f3_10(i,j) < edges(6)  
                        G_f3_10_q(i,j) = new_val_f3_10(5);     
                    elseif G_f3_10(i,j) >= edges(6) && G_f3_10(i,j) < edges(7)  
                        G_f3_10_q(i,j) = new_val_f3_10(6);  
                    elseif G_f3_10(i,j) >= edges(7) && G_f3_10(i,j) < edges(8)  
                        G_f3_10_q(i,j) = new_val_f3_10(7);   
                    elseif G_f3_10(i,j) >= edges(8) && G_f3_10(i,j) < edges(9)  
                        G_f3_10_q(i,j) = new_val_f3_10(8);     
                    elseif G_f3_10(i,j) >= edges(9) && G_f3_10(i,j) < edges(10)  
                        G_f3_10_q(i,j) = new_val_f3_10(9);  
                    elseif G_f3_10(i,j) >= edges(10) && G_f3_10(i,j) < edges(11)  
                        G_f3_10_q(i,j) = new_val_f3_10(10);  
                    elseif G_f3_10(i,j) >= edges(11) && G_f3_10(i,j) < edges(12)  
                        G_f3_10_q(i,j) = new_val_f3_10(11);  
                    elseif G_f3_10(i,j) >= edges(12) && G_f3_10(i,j) < edges(13)  
                        G_f3_10_q(i,j) = new_val_f3_10(12);  
                    elseif G_f3_10(i,j) >= edges(13) && G_f3_10(i,j) < edges(14)   
                         G_f3_10_q(i,j) = new_val_f3_10(13);   
                    elseif G_f3_10(i,j) >= edges(14) && G_f3_10(i,j) < edges(15)  
                        G_f3_10_q(i,j) = new_val_f3_10(14);  
                    elseif G_f3_10(i,j) >= edges(15) && G_f3_10(i,j) < edges(16)  
                        G_f3_10_q(i,j) = new_val_f3_10(15);  
                    elseif G_f3_10(i,j) >= edges(16) && G_f3_10(i,j) <= edges(17)   
                         G_f3_10_q(i,j) = new_val_f3_10(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_11,1) 
                for j = 1:size(G_f3_11,2) 
                    if G_f3_11(i,j) < edges(1)  
                        G_f3_11_q(i,j) = new_val_f3_11(1);  
                    elseif G_f3_11(i,j) >= edges(1) && G_f3_11(i,j) < edges(2)  
                        G_f3_11_q(i,j) = new_val_f3_11(1);  
                    elseif G_f3_11(i,j) >= edges(2) && G_f3_11(i,j) < edges(3)  
                        G_f3_11_q(i,j) = new_val_f3_11(2);    
                    elseif G_f3_11(i,j) >= edges(3) && G_f3_11(i,j) < edges(4)  
                        G_f3_11_q(i,j) = new_val_f3_11(3);  
                    elseif G_f3_11(i,j) >= edges(4) && G_f3_11(i,j) < edges(5)  
                        G_f3_11_q(i,j) = new_val_f3_11(4);   
                    elseif G_f3_11(i,j) >= edges(5) && G_f3_11(i,j) < edges(6)  
                        G_f3_11_q(i,j) = new_val_f3_11(5);     
                    elseif G_f3_11(i,j) >= edges(6) && G_f3_11(i,j) < edges(7)  
                        G_f3_11_q(i,j) = new_val_f3_11(6);  
                    elseif G_f3_11(i,j) >= edges(7) && G_f3_11(i,j) < edges(8)  
                        G_f3_11_q(i,j) = new_val_f3_11(7);   
                    elseif G_f3_11(i,j) >= edges(8) && G_f3_11(i,j) < edges(9)  
                        G_f3_11_q(i,j) = new_val_f3_11(8);     
                    elseif G_f3_11(i,j) >= edges(9) && G_f3_11(i,j) < edges(10)  
                        G_f3_11_q(i,j) = new_val_f3_11(9);  
                    elseif G_f3_11(i,j) >= edges(10) && G_f3_11(i,j) < edges(11)  
                        G_f3_11_q(i,j) = new_val_f3_11(10);  
                    elseif G_f3_11(i,j) >= edges(11) && G_f3_11(i,j) < edges(12)  
                        G_f3_11_q(i,j) = new_val_f3_11(11);  
                    elseif G_f3_11(i,j) >= edges(12) && G_f3_11(i,j) < edges(13)  
                        G_f3_11_q(i,j) = new_val_f3_11(12);  
                    elseif G_f3_11(i,j) >= edges(13) && G_f3_11(i,j) < edges(14)   
                         G_f3_11_q(i,j) = new_val_f3_11(13);   
                    elseif G_f3_11(i,j) >= edges(14) && G_f3_11(i,j) < edges(15)  
                        G_f3_11_q(i,j) = new_val_f3_11(14);  
                    elseif G_f3_11(i,j) >= edges(15) && G_f3_11(i,j) < edges(16)  
                        G_f3_11_q(i,j) = new_val_f3_11(15);  
                    elseif G_f3_11(i,j) >= edges(16) && G_f3_11(i,j) <= edges(17)   
                         G_f3_11_q(i,j) = new_val_f3_11(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_12,1) 
                for j = 1:size(G_f3_12,2) 
                    if G_f3_12(i,j) < edges(1)  
                        G_f3_12_q(i,j) = new_val_f3_12(1);  
                    elseif G_f3_12(i,j) >= edges(1) && G_f3_12(i,j) < edges(2)  
                        G_f3_12_q(i,j) = new_val_f3_12(1);  
                    elseif G_f3_12(i,j) >= edges(2) && G_f3_12(i,j) < edges(3)  
                        G_f3_12_q(i,j) = new_val_f3_12(2);    
                    elseif G_f3_12(i,j) >= edges(3) && G_f3_12(i,j) < edges(4)  
                        G_f3_12_q(i,j) = new_val_f3_12(3);  
                    elseif G_f3_12(i,j) >= edges(4) && G_f3_12(i,j) < edges(5)  
                        G_f3_12_q(i,j) = new_val_f3_12(4);   
                    elseif G_f3_12(i,j) >= edges(5) && G_f3_12(i,j) < edges(6)  
                        G_f3_12_q(i,j) = new_val_f3_12(5);     
                    elseif G_f3_12(i,j) >= edges(6) && G_f3_12(i,j) < edges(7)  
                        G_f3_12_q(i,j) = new_val_f3_12(6);  
                    elseif G_f3_12(i,j) >= edges(7) && G_f3_12(i,j) < edges(8)  
                        G_f3_12_q(i,j) = new_val_f3_12(7);   
                    elseif G_f3_12(i,j) >= edges(8) && G_f3_12(i,j) < edges(9)  
                        G_f3_12_q(i,j) = new_val_f3_12(8);     
                    elseif G_f3_12(i,j) >= edges(9) && G_f3_12(i,j) < edges(10)  
                        G_f3_12_q(i,j) = new_val_f3_12(9);  
                    elseif G_f3_12(i,j) >= edges(10) && G_f3_12(i,j) < edges(11)  
                        G_f3_12_q(i,j) = new_val_f3_12(10);  
                    elseif G_f3_12(i,j) >= edges(11) && G_f3_12(i,j) < edges(12)  
                        G_f3_12_q(i,j) = new_val_f3_12(11);  
                    elseif G_f3_12(i,j) >= edges(12) && G_f3_12(i,j) < edges(13)  
                        G_f3_12_q(i,j) = new_val_f3_12(12);  
                    elseif G_f3_12(i,j) >= edges(13) && G_f3_12(i,j) < edges(14)   
                         G_f3_12_q(i,j) = new_val_f3_12(13);   
                    elseif G_f3_12(i,j) >= edges(14) && G_f3_12(i,j) < edges(15)  
                        G_f3_12_q(i,j) = new_val_f3_12(14);  
                    elseif G_f3_12(i,j) >= edges(15) && G_f3_12(i,j) < edges(16)  
                        G_f3_12_q(i,j) = new_val_f3_12(15);  
                    elseif G_f3_12(i,j) >= edges(16) && G_f3_12(i,j) <= edges(17)   
                         G_f3_12_q(i,j) = new_val_f3_12(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_13,1) 
                for j = 1:size(G_f3_13,2) 
                    if G_f3_13(i,j) < edges(1)  
                        G_f3_13_q(i,j) = new_val_f3_13(1);  
                    elseif G_f3_13(i,j) >= edges(1) && G_f3_13(i,j) < edges(2)  
                        G_f3_13_q(i,j) = new_val_f3_13(1);  
                    elseif G_f3_13(i,j) >= edges(2) && G_f3_13(i,j) < edges(3)  
                        G_f3_13_q(i,j) = new_val_f3_13(2);    
                    elseif G_f3_13(i,j) >= edges(3) && G_f3_13(i,j) < edges(4)  
                        G_f3_13_q(i,j) = new_val_f3_13(3);  
                    elseif G_f3_13(i,j) >= edges(4) && G_f3_13(i,j) < edges(5)  
                        G_f3_13_q(i,j) = new_val_f3_13(4);   
                    elseif G_f3_13(i,j) >= edges(5) && G_f3_13(i,j) < edges(6)  
                        G_f3_13_q(i,j) = new_val_f3_13(5);     
                    elseif G_f3_13(i,j) >= edges(6) && G_f3_13(i,j) < edges(7)  
                        G_f3_13_q(i,j) = new_val_f3_13(6);  
                    elseif G_f3_13(i,j) >= edges(7) && G_f3_13(i,j) < edges(8)  
                        G_f3_13_q(i,j) = new_val_f3_13(7);   
                    elseif G_f3_13(i,j) >= edges(8) && G_f3_13(i,j) < edges(9)  
                        G_f3_13_q(i,j) = new_val_f3_13(8);     
                    elseif G_f3_13(i,j) >= edges(9) && G_f3_13(i,j) < edges(10)  
                        G_f3_13_q(i,j) = new_val_f3_13(9);  
                    elseif G_f3_13(i,j) >= edges(10) && G_f3_13(i,j) < edges(11)  
                        G_f3_13_q(i,j) = new_val_f3_13(10);  
                    elseif G_f3_13(i,j) >= edges(11) && G_f3_13(i,j) < edges(12)  
                        G_f3_13_q(i,j) = new_val_f3_13(11);  
                    elseif G_f3_13(i,j) >= edges(12) && G_f3_13(i,j) < edges(13)  
                        G_f3_13_q(i,j) = new_val_f3_13(12);  
                    elseif G_f3_13(i,j) >= edges(13) && G_f3_13(i,j) < edges(14)   
                         G_f3_13_q(i,j) = new_val_f3_13(13);   
                    elseif G_f3_13(i,j) >= edges(14) && G_f3_13(i,j) < edges(15)  
                        G_f3_13_q(i,j) = new_val_f3_13(14);  
                    elseif G_f3_13(i,j) >= edges(15) && G_f3_13(i,j) < edges(16)  
                        G_f3_13_q(i,j) = new_val_f3_13(15);  
                    elseif G_f3_13(i,j) >= edges(16) && G_f3_13(i,j) <= edges(17)   
                         G_f3_13_q(i,j) = new_val_f3_13(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_14,1) 
                for j = 1:size(G_f3_14,2) 
                    if G_f3_14(i,j) < edges(1)  
                        G_f3_14_q(i,j) = new_val_f3_14(1);  
                    elseif G_f3_14(i,j) >= edges(1) && G_f3_14(i,j) < edges(2)  
                        G_f3_14_q(i,j) = new_val_f3_14(1);  
                    elseif G_f3_14(i,j) >= edges(2) && G_f3_14(i,j) < edges(3)  
                        G_f3_14_q(i,j) = new_val_f3_14(2);    
                    elseif G_f3_14(i,j) >= edges(3) && G_f3_14(i,j) < edges(4)  
                        G_f3_14_q(i,j) = new_val_f3_14(3);  
                    elseif G_f3_14(i,j) >= edges(4) && G_f3_14(i,j) < edges(5)  
                        G_f3_14_q(i,j) = new_val_f3_14(4);   
                    elseif G_f3_14(i,j) >= edges(5) && G_f3_14(i,j) < edges(6)  
                        G_f3_14_q(i,j) = new_val_f3_14(5);     
                    elseif G_f3_14(i,j) >= edges(6) && G_f3_14(i,j) < edges(7)  
                        G_f3_14_q(i,j) = new_val_f3_14(6);  
                    elseif G_f3_14(i,j) >= edges(7) && G_f3_14(i,j) < edges(8)  
                        G_f3_14_q(i,j) = new_val_f3_14(7);   
                    elseif G_f3_14(i,j) >= edges(8) && G_f3_14(i,j) < edges(9)  
                        G_f3_14_q(i,j) = new_val_f3_14(8);     
                    elseif G_f3_14(i,j) >= edges(9) && G_f3_14(i,j) < edges(10)  
                        G_f3_14_q(i,j) = new_val_f3_14(9);  
                    elseif G_f3_14(i,j) >= edges(10) && G_f3_14(i,j) < edges(11)  
                        G_f3_14_q(i,j) = new_val_f3_14(10);  
                    elseif G_f3_14(i,j) >= edges(11) && G_f3_14(i,j) < edges(12)  
                        G_f3_14_q(i,j) = new_val_f3_14(11);  
                    elseif G_f3_14(i,j) >= edges(12) && G_f3_14(i,j) < edges(13)  
                        G_f3_14_q(i,j) = new_val_f3_14(12);  
                    elseif G_f3_14(i,j) >= edges(13) && G_f3_14(i,j) < edges(14)   
                         G_f3_14_q(i,j) = new_val_f3_14(13);   
                    elseif G_f3_14(i,j) >= edges(14) && G_f3_14(i,j) < edges(15)  
                        G_f3_14_q(i,j) = new_val_f3_14(14);  
                    elseif G_f3_14(i,j) >= edges(15) && G_f3_14(i,j) < edges(16)  
                        G_f3_14_q(i,j) = new_val_f3_14(15);  
                    elseif G_f3_14(i,j) >= edges(16) && G_f3_14(i,j) <= edges(17)   
                         G_f3_14_q(i,j) = new_val_f3_14(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_15,1) 
                for j = 1:size(G_f3_15,2) 
                    if G_f3_15(i,j) < edges(1)  
                        G_f3_15_q(i,j) = new_val_f3_15(1);  
                    elseif G_f3_15(i,j) >= edges(1) && G_f3_15(i,j) < edges(2)  
                        G_f3_15_q(i,j) = new_val_f3_15(1);  
                    elseif G_f3_15(i,j) >= edges(2) && G_f3_15(i,j) < edges(3)  
                        G_f3_15_q(i,j) = new_val_f3_15(2);    
                    elseif G_f3_15(i,j) >= edges(3) && G_f3_15(i,j) < edges(4)  
                        G_f3_15_q(i,j) = new_val_f3_15(3);  
                    elseif G_f3_15(i,j) >= edges(4) && G_f3_15(i,j) < edges(5)  
                        G_f3_15_q(i,j) = new_val_f3_15(4);   
                    elseif G_f3_15(i,j) >= edges(5) && G_f3_15(i,j) < edges(6)  
                        G_f3_15_q(i,j) = new_val_f3_15(5);     
                    elseif G_f3_15(i,j) >= edges(6) && G_f3_15(i,j) < edges(7)  
                        G_f3_15_q(i,j) = new_val_f3_15(6);  
                    elseif G_f3_15(i,j) >= edges(7) && G_f3_15(i,j) < edges(8)  
                        G_f3_15_q(i,j) = new_val_f3_15(7);   
                    elseif G_f3_15(i,j) >= edges(8) && G_f3_15(i,j) < edges(9)  
                        G_f3_15_q(i,j) = new_val_f3_15(8);     
                    elseif G_f3_15(i,j) >= edges(9) && G_f3_15(i,j) < edges(10)  
                        G_f3_15_q(i,j) = new_val_f3_15(9);  
                    elseif G_f3_15(i,j) >= edges(10) && G_f3_15(i,j) < edges(11)  
                        G_f3_15_q(i,j) = new_val_f3_15(10);  
                    elseif G_f3_15(i,j) >= edges(11) && G_f3_15(i,j) < edges(12)  
                        G_f3_15_q(i,j) = new_val_f3_15(11);  
                    elseif G_f3_15(i,j) >= edges(12) && G_f3_15(i,j) < edges(13)  
                        G_f3_15_q(i,j) = new_val_f3_15(12);  
                    elseif G_f3_15(i,j) >= edges(13) && G_f3_15(i,j) < edges(14)   
                         G_f3_15_q(i,j) = new_val_f3_15(13);   
                    elseif G_f3_15(i,j) >= edges(14) && G_f3_15(i,j) < edges(15)  
                        G_f3_15_q(i,j) = new_val_f3_15(14);  
                    elseif G_f3_15(i,j) >= edges(15) && G_f3_15(i,j) < edges(16)  
                        G_f3_15_q(i,j) = new_val_f3_15(15);  
                    elseif G_f3_15(i,j) >= edges(16) && G_f3_15(i,j) <= edges(17)   
                         G_f3_15_q(i,j) = new_val_f3_15(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_16,1) 
                for j = 1:size(G_f3_16,2) 
                    if G_f3_16(i,j) < edges(1)  
                        G_f3_16_q(i,j) = new_val_f3_16(1);  
                    elseif G_f3_16(i,j) >= edges(1) && G_f3_16(i,j) < edges(2)  
                        G_f3_16_q(i,j) = new_val_f3_16(1);  
                    elseif G_f3_16(i,j) >= edges(2) && G_f3_16(i,j) < edges(3)  
                        G_f3_16_q(i,j) = new_val_f3_16(2);    
                    elseif G_f3_16(i,j) >= edges(3) && G_f3_16(i,j) < edges(4)  
                        G_f3_16_q(i,j) = new_val_f3_16(3);  
                    elseif G_f3_16(i,j) >= edges(4) && G_f3_16(i,j) < edges(5)  
                        G_f3_16_q(i,j) = new_val_f3_16(4);   
                    elseif G_f3_16(i,j) >= edges(5) && G_f3_16(i,j) < edges(6)  
                        G_f3_16_q(i,j) = new_val_f3_16(5);     
                    elseif G_f3_16(i,j) >= edges(6) && G_f3_16(i,j) < edges(7)  
                        G_f3_16_q(i,j) = new_val_f3_16(6);  
                    elseif G_f3_16(i,j) >= edges(7) && G_f3_16(i,j) < edges(8)  
                        G_f3_16_q(i,j) = new_val_f3_16(7);   
                    elseif G_f3_16(i,j) >= edges(8) && G_f3_16(i,j) < edges(9)  
                        G_f3_16_q(i,j) = new_val_f3_16(8);     
                    elseif G_f3_16(i,j) >= edges(9) && G_f3_16(i,j) < edges(10)  
                        G_f3_16_q(i,j) = new_val_f3_16(9);  
                    elseif G_f3_16(i,j) >= edges(10) && G_f3_16(i,j) < edges(11)  
                        G_f3_16_q(i,j) = new_val_f3_16(10);  
                    elseif G_f3_16(i,j) >= edges(11) && G_f3_16(i,j) < edges(12)  
                        G_f3_16_q(i,j) = new_val_f3_16(11);  
                    elseif G_f3_16(i,j) >= edges(12) && G_f3_16(i,j) < edges(13)  
                        G_f3_16_q(i,j) = new_val_f3_16(12);  
                    elseif G_f3_16(i,j) >= edges(13) && G_f3_16(i,j) < edges(14)   
                         G_f3_16_q(i,j) = new_val_f3_16(13);   
                    elseif G_f3_16(i,j) >= edges(14) && G_f3_16(i,j) < edges(15)  
                        G_f3_16_q(i,j) = new_val_f3_16(14);  
                    elseif G_f3_16(i,j) >= edges(15) && G_f3_16(i,j) < edges(16)  
                        G_f3_16_q(i,j) = new_val_f3_16(15);  
                    elseif G_f3_16(i,j) >= edges(16) && G_f3_16(i,j) <= edges(17)   
                         G_f3_16_q(i,j) = new_val_f3_16(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_17,1) 
                for j = 1:size(G_f3_17,2) 
                    if G_f3_17(i,j) < edges(1)  
                        G_f3_17_q(i,j) = new_val_f3_17(1);  
                    elseif G_f3_17(i,j) >= edges(1) && G_f3_17(i,j) < edges(2)  
                        G_f3_17_q(i,j) = new_val_f3_17(1);  
                    elseif G_f3_17(i,j) >= edges(2) && G_f3_17(i,j) < edges(3)  
                        G_f3_17_q(i,j) = new_val_f3_17(2);    
                    elseif G_f3_17(i,j) >= edges(3) && G_f3_17(i,j) < edges(4)  
                        G_f3_17_q(i,j) = new_val_f3_17(3);  
                    elseif G_f3_17(i,j) >= edges(4) && G_f3_17(i,j) < edges(5)  
                        G_f3_17_q(i,j) = new_val_f3_17(4);   
                    elseif G_f3_17(i,j) >= edges(5) && G_f3_17(i,j) < edges(6)  
                        G_f3_17_q(i,j) = new_val_f3_17(5);     
                    elseif G_f3_17(i,j) >= edges(6) && G_f3_17(i,j) < edges(7)  
                        G_f3_17_q(i,j) = new_val_f3_17(6);  
                    elseif G_f3_17(i,j) >= edges(7) && G_f3_17(i,j) < edges(8)  
                        G_f3_17_q(i,j) = new_val_f3_17(7);   
                    elseif G_f3_17(i,j) >= edges(8) && G_f3_17(i,j) < edges(9)  
                        G_f3_17_q(i,j) = new_val_f3_17(8);     
                    elseif G_f3_17(i,j) >= edges(9) && G_f3_17(i,j) < edges(10)  
                        G_f3_17_q(i,j) = new_val_f3_17(9);  
                    elseif G_f3_17(i,j) >= edges(10) && G_f3_17(i,j) < edges(11)  
                        G_f3_17_q(i,j) = new_val_f3_17(10);  
                    elseif G_f3_17(i,j) >= edges(11) && G_f3_17(i,j) < edges(12)  
                        G_f3_17_q(i,j) = new_val_f3_17(11);  
                    elseif G_f3_17(i,j) >= edges(12) && G_f3_17(i,j) < edges(13)  
                        G_f3_17_q(i,j) = new_val_f3_17(12);  
                    elseif G_f3_17(i,j) >= edges(13) && G_f3_17(i,j) < edges(14)   
                         G_f3_17_q(i,j) = new_val_f3_17(13);   
                    elseif G_f3_17(i,j) >= edges(14) && G_f3_17(i,j) < edges(15)  
                        G_f3_17_q(i,j) = new_val_f3_17(14);  
                    elseif G_f3_17(i,j) >= edges(15) && G_f3_17(i,j) < edges(16)  
                        G_f3_17_q(i,j) = new_val_f3_17(15);  
                    elseif G_f3_17(i,j) >= edges(16) && G_f3_17(i,j) <= edges(17)   
                         G_f3_17_q(i,j) = new_val_f3_17(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_18,1) 
                for j = 1:size(G_f3_18,2) 
                    if G_f3_18(i,j) < edges(1)  
                        G_f3_18_q(i,j) = new_val_f3_18(1);  
                    elseif G_f3_18(i,j) >= edges(1) && G_f3_18(i,j) < edges(2)  
                        G_f3_18_q(i,j) = new_val_f3_18(1);  
                    elseif G_f3_18(i,j) >= edges(2) && G_f3_18(i,j) < edges(3)  
                        G_f3_18_q(i,j) = new_val_f3_18(2);    
                    elseif G_f3_18(i,j) >= edges(3) && G_f3_18(i,j) < edges(4)  
                        G_f3_18_q(i,j) = new_val_f3_18(3);  
                    elseif G_f3_18(i,j) >= edges(4) && G_f3_18(i,j) < edges(5)  
                        G_f3_18_q(i,j) = new_val_f3_18(4);   
                    elseif G_f3_18(i,j) >= edges(5) && G_f3_18(i,j) < edges(6)  
                        G_f3_18_q(i,j) = new_val_f3_18(5);     
                    elseif G_f3_18(i,j) >= edges(6) && G_f3_18(i,j) < edges(7)  
                        G_f3_18_q(i,j) = new_val_f3_18(6);  
                    elseif G_f3_18(i,j) >= edges(7) && G_f3_18(i,j) < edges(8)  
                        G_f3_18_q(i,j) = new_val_f3_18(7);   
                    elseif G_f3_18(i,j) >= edges(8) && G_f3_18(i,j) < edges(9)  
                        G_f3_18_q(i,j) = new_val_f3_18(8);     
                    elseif G_f3_18(i,j) >= edges(9) && G_f3_18(i,j) < edges(10)  
                        G_f3_18_q(i,j) = new_val_f3_18(9);  
                    elseif G_f3_18(i,j) >= edges(10) && G_f3_18(i,j) < edges(11)  
                        G_f3_18_q(i,j) = new_val_f3_18(10);  
                    elseif G_f3_18(i,j) >= edges(11) && G_f3_18(i,j) < edges(12)  
                        G_f3_18_q(i,j) = new_val_f3_18(11);  
                    elseif G_f3_18(i,j) >= edges(12) && G_f3_18(i,j) < edges(13)  
                        G_f3_18_q(i,j) = new_val_f3_18(12);  
                    elseif G_f3_18(i,j) >= edges(13) && G_f3_18(i,j) < edges(14)   
                         G_f3_18_q(i,j) = new_val_f3_18(13);   
                    elseif G_f3_18(i,j) >= edges(14) && G_f3_18(i,j) < edges(15)  
                        G_f3_18_q(i,j) = new_val_f3_18(14);  
                    elseif G_f3_18(i,j) >= edges(15) && G_f3_18(i,j) < edges(16)  
                        G_f3_18_q(i,j) = new_val_f3_18(15);  
                    elseif G_f3_18(i,j) >= edges(16) && G_f3_18(i,j) <= edges(17)   
                         G_f3_18_q(i,j) = new_val_f3_18(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_19,1) 
                for j = 1:size(G_f3_19,2) 
                    if G_f3_19(i,j) < edges(1)  
                        G_f3_19_q(i,j) = new_val_f3_19(1);  
                    elseif G_f3_19(i,j) >= edges(1) && G_f3_19(i,j) < edges(2)  
                        G_f3_19_q(i,j) = new_val_f3_19(1);  
                    elseif G_f3_19(i,j) >= edges(2) && G_f3_19(i,j) < edges(3)  
                        G_f3_19_q(i,j) = new_val_f3_19(2);    
                    elseif G_f3_19(i,j) >= edges(3) && G_f3_19(i,j) < edges(4)  
                        G_f3_19_q(i,j) = new_val_f3_19(3);  
                    elseif G_f3_19(i,j) >= edges(4) && G_f3_19(i,j) < edges(5)  
                        G_f3_19_q(i,j) = new_val_f3_19(4);   
                    elseif G_f3_19(i,j) >= edges(5) && G_f3_19(i,j) < edges(6)  
                        G_f3_19_q(i,j) = new_val_f3_19(5);     
                    elseif G_f3_19(i,j) >= edges(6) && G_f3_19(i,j) < edges(7)  
                        G_f3_19_q(i,j) = new_val_f3_19(6);  
                    elseif G_f3_19(i,j) >= edges(7) && G_f3_19(i,j) < edges(8)  
                        G_f3_19_q(i,j) = new_val_f3_19(7);   
                    elseif G_f3_19(i,j) >= edges(8) && G_f3_19(i,j) < edges(9)  
                        G_f3_19_q(i,j) = new_val_f3_19(8);     
                    elseif G_f3_19(i,j) >= edges(9) && G_f3_19(i,j) < edges(10)  
                        G_f3_19_q(i,j) = new_val_f3_19(9);  
                    elseif G_f3_19(i,j) >= edges(10) && G_f3_19(i,j) < edges(11)  
                        G_f3_19_q(i,j) = new_val_f3_19(10);  
                    elseif G_f3_19(i,j) >= edges(11) && G_f3_19(i,j) < edges(12)  
                        G_f3_19_q(i,j) = new_val_f3_19(11);  
                    elseif G_f3_19(i,j) >= edges(12) && G_f3_19(i,j) < edges(13)  
                        G_f3_19_q(i,j) = new_val_f3_19(12);  
                    elseif G_f3_19(i,j) >= edges(13) && G_f3_19(i,j) < edges(14)   
                         G_f3_19_q(i,j) = new_val_f3_19(13);   
                    elseif G_f3_19(i,j) >= edges(14) && G_f3_19(i,j) < edges(15)  
                        G_f3_19_q(i,j) = new_val_f3_19(14);  
                    elseif G_f3_19(i,j) >= edges(15) && G_f3_19(i,j) < edges(16)  
                        G_f3_19_q(i,j) = new_val_f3_19(15);  
                    elseif G_f3_19(i,j) >= edges(16) && G_f3_19(i,j) <= edges(17)   
                         G_f3_19_q(i,j) = new_val_f3_19(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_20,1) 
                for j = 1:size(G_f3_20,2) 
                    if G_f3_20(i,j) < edges(1)  
                        G_f3_20_q(i,j) = new_val_f3_20(1);  
                    elseif G_f3_20(i,j) >= edges(1) && G_f3_20(i,j) < edges(2)  
                        G_f3_20_q(i,j) = new_val_f3_20(1);  
                    elseif G_f3_20(i,j) >= edges(2) && G_f3_20(i,j) < edges(3)  
                        G_f3_20_q(i,j) = new_val_f3_20(2);    
                    elseif G_f3_20(i,j) >= edges(3) && G_f3_20(i,j) < edges(4)  
                        G_f3_20_q(i,j) = new_val_f3_20(3);  
                    elseif G_f3_20(i,j) >= edges(4) && G_f3_20(i,j) < edges(5)  
                        G_f3_20_q(i,j) = new_val_f3_20(4);   
                    elseif G_f3_20(i,j) >= edges(5) && G_f3_20(i,j) < edges(6)  
                        G_f3_20_q(i,j) = new_val_f3_20(5);     
                    elseif G_f3_20(i,j) >= edges(6) && G_f3_20(i,j) < edges(7)  
                        G_f3_20_q(i,j) = new_val_f3_20(6);  
                    elseif G_f3_20(i,j) >= edges(7) && G_f3_20(i,j) < edges(8)  
                        G_f3_20_q(i,j) = new_val_f3_20(7);   
                    elseif G_f3_20(i,j) >= edges(8) && G_f3_20(i,j) < edges(9)  
                        G_f3_20_q(i,j) = new_val_f3_20(8);     
                    elseif G_f3_20(i,j) >= edges(9) && G_f3_20(i,j) < edges(10)  
                        G_f3_20_q(i,j) = new_val_f3_20(9);  
                    elseif G_f3_20(i,j) >= edges(10) && G_f3_20(i,j) < edges(11)  
                        G_f3_20_q(i,j) = new_val_f3_20(10);  
                    elseif G_f3_20(i,j) >= edges(11) && G_f3_20(i,j) < edges(12)  
                        G_f3_20_q(i,j) = new_val_f3_20(11);  
                    elseif G_f3_20(i,j) >= edges(12) && G_f3_20(i,j) < edges(13)  
                        G_f3_20_q(i,j) = new_val_f3_20(12);  
                    elseif G_f3_20(i,j) >= edges(13) && G_f3_20(i,j) < edges(14)   
                         G_f3_20_q(i,j) = new_val_f3_20(13);   
                    elseif G_f3_20(i,j) >= edges(14) && G_f3_20(i,j) < edges(15)  
                        G_f3_20_q(i,j) = new_val_f3_20(14);  
                    elseif G_f3_20(i,j) >= edges(15) && G_f3_20(i,j) < edges(16)  
                        G_f3_20_q(i,j) = new_val_f3_20(15);  
                    elseif G_f3_20(i,j) >= edges(16) && G_f3_20(i,j) <= edges(17)   
                         G_f3_20_q(i,j) = new_val_f3_20(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_21,1) 
                for j = 1:size(G_f3_21,2) 
                    if G_f3_21(i,j) < edges(1)  
                        G_f3_21_q(i,j) = new_val_f3_21(1);  
                    elseif G_f3_21(i,j) >= edges(1) && G_f3_21(i,j) < edges(2)  
                        G_f3_21_q(i,j) = new_val_f3_21(1);  
                    elseif G_f3_21(i,j) >= edges(2) && G_f3_21(i,j) < edges(3)  
                        G_f3_21_q(i,j) = new_val_f3_21(2);    
                    elseif G_f3_21(i,j) >= edges(3) && G_f3_21(i,j) < edges(4)  
                        G_f3_21_q(i,j) = new_val_f3_21(3);  
                    elseif G_f3_21(i,j) >= edges(4) && G_f3_21(i,j) < edges(5)  
                        G_f3_21_q(i,j) = new_val_f3_21(4);   
                    elseif G_f3_21(i,j) >= edges(5) && G_f3_21(i,j) < edges(6)  
                        G_f3_21_q(i,j) = new_val_f3_21(5);     
                    elseif G_f3_21(i,j) >= edges(6) && G_f3_21(i,j) < edges(7)  
                        G_f3_21_q(i,j) = new_val_f3_21(6);  
                    elseif G_f3_21(i,j) >= edges(7) && G_f3_21(i,j) < edges(8)  
                        G_f3_21_q(i,j) = new_val_f3_21(7);   
                    elseif G_f3_21(i,j) >= edges(8) && G_f3_21(i,j) < edges(9)  
                        G_f3_21_q(i,j) = new_val_f3_21(8);     
                    elseif G_f3_21(i,j) >= edges(9) && G_f3_21(i,j) < edges(10)  
                        G_f3_21_q(i,j) = new_val_f3_21(9);  
                    elseif G_f3_21(i,j) >= edges(10) && G_f3_21(i,j) < edges(11)  
                        G_f3_21_q(i,j) = new_val_f3_21(10);  
                    elseif G_f3_21(i,j) >= edges(11) && G_f3_21(i,j) < edges(12)  
                        G_f3_21_q(i,j) = new_val_f3_21(11);  
                    elseif G_f3_21(i,j) >= edges(12) && G_f3_21(i,j) < edges(13)  
                        G_f3_21_q(i,j) = new_val_f3_21(12);  
                    elseif G_f3_21(i,j) >= edges(13) && G_f3_21(i,j) < edges(14)   
                         G_f3_21_q(i,j) = new_val_f3_21(13);   
                    elseif G_f3_21(i,j) >= edges(14) && G_f3_21(i,j) < edges(15)  
                        G_f3_21_q(i,j) = new_val_f3_21(14);  
                    elseif G_f3_21(i,j) >= edges(15) && G_f3_21(i,j) < edges(16)  
                        G_f3_21_q(i,j) = new_val_f3_21(15);  
                    elseif G_f3_21(i,j) >= edges(16) && G_f3_21(i,j) <= edges(17)   
                         G_f3_21_q(i,j) = new_val_f3_21(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_22,1) 
                for j = 1:size(G_f3_22,2) 
                    if G_f3_22(i,j) < edges(1)  
                        G_f3_22_q(i,j) = new_val_f3_22(1);  
                    elseif G_f3_22(i,j) >= edges(1) && G_f3_22(i,j) < edges(2)  
                        G_f3_22_q(i,j) = new_val_f3_22(1);  
                    elseif G_f3_22(i,j) >= edges(2) && G_f3_22(i,j) < edges(3)  
                        G_f3_22_q(i,j) = new_val_f3_22(2);    
                    elseif G_f3_22(i,j) >= edges(3) && G_f3_22(i,j) < edges(4)  
                        G_f3_22_q(i,j) = new_val_f3_22(3);  
                    elseif G_f3_22(i,j) >= edges(4) && G_f3_22(i,j) < edges(5)  
                        G_f3_22_q(i,j) = new_val_f3_22(4);   
                    elseif G_f3_22(i,j) >= edges(5) && G_f3_22(i,j) < edges(6)  
                        G_f3_22_q(i,j) = new_val_f3_22(5);     
                    elseif G_f3_22(i,j) >= edges(6) && G_f3_22(i,j) < edges(7)  
                        G_f3_22_q(i,j) = new_val_f3_22(6);  
                    elseif G_f3_22(i,j) >= edges(7) && G_f3_22(i,j) < edges(8)  
                        G_f3_22_q(i,j) = new_val_f3_22(7);   
                    elseif G_f3_22(i,j) >= edges(8) && G_f3_22(i,j) < edges(9)  
                        G_f3_22_q(i,j) = new_val_f3_22(8);     
                    elseif G_f3_22(i,j) >= edges(9) && G_f3_22(i,j) < edges(10)  
                        G_f3_22_q(i,j) = new_val_f3_22(9);  
                    elseif G_f3_22(i,j) >= edges(10) && G_f3_22(i,j) < edges(11)  
                        G_f3_22_q(i,j) = new_val_f3_22(10);  
                    elseif G_f3_22(i,j) >= edges(11) && G_f3_22(i,j) < edges(12)  
                        G_f3_22_q(i,j) = new_val_f3_22(11);  
                    elseif G_f3_22(i,j) >= edges(12) && G_f3_22(i,j) < edges(13)  
                        G_f3_22_q(i,j) = new_val_f3_22(12);  
                    elseif G_f3_22(i,j) >= edges(13) && G_f3_22(i,j) < edges(14)   
                         G_f3_22_q(i,j) = new_val_f3_22(13);   
                    elseif G_f3_22(i,j) >= edges(14) && G_f3_22(i,j) < edges(15)  
                        G_f3_22_q(i,j) = new_val_f3_22(14);  
                    elseif G_f3_22(i,j) >= edges(15) && G_f3_22(i,j) < edges(16)  
                        G_f3_22_q(i,j) = new_val_f3_22(15);  
                    elseif G_f3_22(i,j) >= edges(16) && G_f3_22(i,j) <= edges(17)   
                         G_f3_22_q(i,j) = new_val_f3_22(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_23,1) 
                for j = 1:size(G_f3_23,2) 
                    if G_f3_23(i,j) < edges(1)  
                        G_f3_23_q(i,j) = new_val_f3_23(1);  
                    elseif G_f3_23(i,j) >= edges(1) && G_f3_23(i,j) < edges(2)  
                        G_f3_23_q(i,j) = new_val_f3_23(1);  
                    elseif G_f3_23(i,j) >= edges(2) && G_f3_23(i,j) < edges(3)  
                        G_f3_23_q(i,j) = new_val_f3_23(2);    
                    elseif G_f3_23(i,j) >= edges(3) && G_f3_23(i,j) < edges(4)  
                        G_f3_23_q(i,j) = new_val_f3_23(3);  
                    elseif G_f3_23(i,j) >= edges(4) && G_f3_23(i,j) < edges(5)  
                        G_f3_23_q(i,j) = new_val_f3_23(4);   
                    elseif G_f3_23(i,j) >= edges(5) && G_f3_23(i,j) < edges(6)  
                        G_f3_23_q(i,j) = new_val_f3_23(5);     
                    elseif G_f3_23(i,j) >= edges(6) && G_f3_23(i,j) < edges(7)  
                        G_f3_23_q(i,j) = new_val_f3_23(6);  
                    elseif G_f3_23(i,j) >= edges(7) && G_f3_23(i,j) < edges(8)  
                        G_f3_23_q(i,j) = new_val_f3_23(7);   
                    elseif G_f3_23(i,j) >= edges(8) && G_f3_23(i,j) < edges(9)  
                        G_f3_23_q(i,j) = new_val_f3_23(8);     
                    elseif G_f3_23(i,j) >= edges(9) && G_f3_23(i,j) < edges(10)  
                        G_f3_23_q(i,j) = new_val_f3_23(9);  
                    elseif G_f3_23(i,j) >= edges(10) && G_f3_23(i,j) < edges(11)  
                        G_f3_23_q(i,j) = new_val_f3_23(10);  
                    elseif G_f3_23(i,j) >= edges(11) && G_f3_23(i,j) < edges(12)  
                        G_f3_23_q(i,j) = new_val_f3_23(11);  
                    elseif G_f3_23(i,j) >= edges(12) && G_f3_23(i,j) < edges(13)  
                        G_f3_23_q(i,j) = new_val_f3_23(12);  
                    elseif G_f3_23(i,j) >= edges(13) && G_f3_23(i,j) < edges(14)   
                         G_f3_23_q(i,j) = new_val_f3_23(13);   
                    elseif G_f3_23(i,j) >= edges(14) && G_f3_23(i,j) < edges(15)  
                        G_f3_23_q(i,j) = new_val_f3_23(14);  
                    elseif G_f3_23(i,j) >= edges(15) && G_f3_23(i,j) < edges(16)  
                        G_f3_23_q(i,j) = new_val_f3_23(15);  
                    elseif G_f3_23(i,j) >= edges(16) && G_f3_23(i,j) <= edges(17)   
                         G_f3_23_q(i,j) = new_val_f3_23(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_24,1) 
                for j = 1:size(G_f3_24,2) 
                    if G_f3_24(i,j) < edges(1)  
                        G_f3_24_q(i,j) = new_val_f3_24(1);  
                    elseif G_f3_24(i,j) >= edges(1) && G_f3_24(i,j) < edges(2)  
                        G_f3_24_q(i,j) = new_val_f3_24(1);  
                    elseif G_f3_24(i,j) >= edges(2) && G_f3_24(i,j) < edges(3)  
                        G_f3_24_q(i,j) = new_val_f3_24(2);    
                    elseif G_f3_24(i,j) >= edges(3) && G_f3_24(i,j) < edges(4)  
                        G_f3_24_q(i,j) = new_val_f3_24(3);  
                    elseif G_f3_24(i,j) >= edges(4) && G_f3_24(i,j) < edges(5)  
                        G_f3_24_q(i,j) = new_val_f3_24(4);   
                    elseif G_f3_24(i,j) >= edges(5) && G_f3_24(i,j) < edges(6)  
                        G_f3_24_q(i,j) = new_val_f3_24(5);     
                    elseif G_f3_24(i,j) >= edges(6) && G_f3_24(i,j) < edges(7)  
                        G_f3_24_q(i,j) = new_val_f3_24(6);  
                    elseif G_f3_24(i,j) >= edges(7) && G_f3_24(i,j) < edges(8)  
                        G_f3_24_q(i,j) = new_val_f3_24(7);   
                    elseif G_f3_24(i,j) >= edges(8) && G_f3_24(i,j) < edges(9)  
                        G_f3_24_q(i,j) = new_val_f3_24(8);     
                    elseif G_f3_24(i,j) >= edges(9) && G_f3_24(i,j) < edges(10)  
                        G_f3_24_q(i,j) = new_val_f3_24(9);  
                    elseif G_f3_24(i,j) >= edges(10) && G_f3_24(i,j) < edges(11)  
                        G_f3_24_q(i,j) = new_val_f3_24(10);  
                    elseif G_f3_24(i,j) >= edges(11) && G_f3_24(i,j) < edges(12)  
                        G_f3_24_q(i,j) = new_val_f3_24(11);  
                    elseif G_f3_24(i,j) >= edges(12) && G_f3_24(i,j) < edges(13)  
                        G_f3_24_q(i,j) = new_val_f3_24(12);  
                    elseif G_f3_24(i,j) >= edges(13) && G_f3_24(i,j) < edges(14)   
                         G_f3_24_q(i,j) = new_val_f3_24(13);   
                    elseif G_f3_24(i,j) >= edges(14) && G_f3_24(i,j) < edges(15)  
                        G_f3_24_q(i,j) = new_val_f3_24(14);  
                    elseif G_f3_24(i,j) >= edges(15) && G_f3_24(i,j) < edges(16)  
                        G_f3_24_q(i,j) = new_val_f3_24(15);  
                    elseif G_f3_24(i,j) >= edges(16) && G_f3_24(i,j) <= edges(17)   
                         G_f3_24_q(i,j) = new_val_f3_24(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_25,1) 
                for j = 1:size(G_f3_25,2) 
                    if G_f3_25(i,j) < edges(1)  
                        G_f3_25_q(i,j) = new_val_f3_25(1);  
                    elseif G_f3_25(i,j) >= edges(1) && G_f3_25(i,j) < edges(2)  
                        G_f3_25_q(i,j) = new_val_f3_25(1);  
                    elseif G_f3_25(i,j) >= edges(2) && G_f3_25(i,j) < edges(3)  
                        G_f3_25_q(i,j) = new_val_f3_25(2);    
                    elseif G_f3_25(i,j) >= edges(3) && G_f3_25(i,j) < edges(4)  
                        G_f3_25_q(i,j) = new_val_f3_25(3);  
                    elseif G_f3_25(i,j) >= edges(4) && G_f3_25(i,j) < edges(5)  
                        G_f3_25_q(i,j) = new_val_f3_25(4);   
                    elseif G_f3_25(i,j) >= edges(5) && G_f3_25(i,j) < edges(6)  
                        G_f3_25_q(i,j) = new_val_f3_25(5);     
                    elseif G_f3_25(i,j) >= edges(6) && G_f3_25(i,j) < edges(7)  
                        G_f3_25_q(i,j) = new_val_f3_25(6);  
                    elseif G_f3_25(i,j) >= edges(7) && G_f3_25(i,j) < edges(8)  
                        G_f3_25_q(i,j) = new_val_f3_25(7);   
                    elseif G_f3_25(i,j) >= edges(8) && G_f3_25(i,j) < edges(9)  
                        G_f3_25_q(i,j) = new_val_f3_25(8);     
                    elseif G_f3_25(i,j) >= edges(9) && G_f3_25(i,j) < edges(10)  
                        G_f3_25_q(i,j) = new_val_f3_25(9);  
                    elseif G_f3_25(i,j) >= edges(10) && G_f3_25(i,j) < edges(11)  
                        G_f3_25_q(i,j) = new_val_f3_25(10);  
                    elseif G_f3_25(i,j) >= edges(11) && G_f3_25(i,j) < edges(12)  
                        G_f3_25_q(i,j) = new_val_f3_25(11);  
                    elseif G_f3_25(i,j) >= edges(12) && G_f3_25(i,j) < edges(13)  
                        G_f3_25_q(i,j) = new_val_f3_25(12);  
                    elseif G_f3_25(i,j) >= edges(13) && G_f3_25(i,j) < edges(14)   
                         G_f3_25_q(i,j) = new_val_f3_25(13);   
                    elseif G_f3_25(i,j) >= edges(14) && G_f3_25(i,j) < edges(15)  
                        G_f3_25_q(i,j) = new_val_f3_25(14);  
                    elseif G_f3_25(i,j) >= edges(15) && G_f3_25(i,j) < edges(16)  
                        G_f3_25_q(i,j) = new_val_f3_25(15);  
                    elseif G_f3_25(i,j) >= edges(16) && G_f3_25(i,j) <= edges(17)   
                         G_f3_25_q(i,j) = new_val_f3_25(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_26,1) 
                for j = 1:size(G_f3_26,2) 
                    if G_f3_26(i,j) < edges(1)  
                        G_f3_26_q(i,j) = new_val_f3_26(1);  
                    elseif G_f3_26(i,j) >= edges(1) && G_f3_26(i,j) < edges(2)  
                        G_f3_26_q(i,j) = new_val_f3_26(1);  
                    elseif G_f3_26(i,j) >= edges(2) && G_f3_26(i,j) < edges(3)  
                        G_f3_26_q(i,j) = new_val_f3_26(2);    
                    elseif G_f3_26(i,j) >= edges(3) && G_f3_26(i,j) < edges(4)  
                        G_f3_26_q(i,j) = new_val_f3_26(3);  
                    elseif G_f3_26(i,j) >= edges(4) && G_f3_26(i,j) < edges(5)  
                        G_f3_26_q(i,j) = new_val_f3_26(4);   
                    elseif G_f3_26(i,j) >= edges(5) && G_f3_26(i,j) < edges(6)  
                        G_f3_26_q(i,j) = new_val_f3_26(5);     
                    elseif G_f3_26(i,j) >= edges(6) && G_f3_26(i,j) < edges(7)  
                        G_f3_26_q(i,j) = new_val_f3_26(6);  
                    elseif G_f3_26(i,j) >= edges(7) && G_f3_26(i,j) < edges(8)  
                        G_f3_26_q(i,j) = new_val_f3_26(7);   
                    elseif G_f3_26(i,j) >= edges(8) && G_f3_26(i,j) < edges(9)  
                        G_f3_26_q(i,j) = new_val_f3_26(8);     
                    elseif G_f3_26(i,j) >= edges(9) && G_f3_26(i,j) < edges(10)  
                        G_f3_26_q(i,j) = new_val_f3_26(9);  
                    elseif G_f3_26(i,j) >= edges(10) && G_f3_26(i,j) < edges(11)  
                        G_f3_26_q(i,j) = new_val_f3_26(10);  
                    elseif G_f3_26(i,j) >= edges(11) && G_f3_26(i,j) < edges(12)  
                        G_f3_26_q(i,j) = new_val_f3_26(11);  
                    elseif G_f3_26(i,j) >= edges(12) && G_f3_26(i,j) < edges(13)  
                        G_f3_26_q(i,j) = new_val_f3_26(12);  
                    elseif G_f3_26(i,j) >= edges(13) && G_f3_26(i,j) < edges(14)   
                         G_f3_26_q(i,j) = new_val_f3_26(13);   
                    elseif G_f3_26(i,j) >= edges(14) && G_f3_26(i,j) < edges(15)  
                        G_f3_26_q(i,j) = new_val_f3_26(14);  
                    elseif G_f3_26(i,j) >= edges(15) && G_f3_26(i,j) < edges(16)  
                        G_f3_26_q(i,j) = new_val_f3_26(15);  
                    elseif G_f3_26(i,j) >= edges(16) && G_f3_26(i,j) <= edges(17)   
                         G_f3_26_q(i,j) = new_val_f3_26(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_27,1) 
                for j = 1:size(G_f3_27,2) 
                    if G_f3_27(i,j) < edges(1)  
                        G_f3_27_q(i,j) = new_val_f3_27(1);  
                    elseif G_f3_27(i,j) >= edges(1) && G_f3_27(i,j) < edges(2)  
                        G_f3_27_q(i,j) = new_val_f3_27(1);  
                    elseif G_f3_27(i,j) >= edges(2) && G_f3_27(i,j) < edges(3)  
                        G_f3_27_q(i,j) = new_val_f3_27(2);    
                    elseif G_f3_27(i,j) >= edges(3) && G_f3_27(i,j) < edges(4)  
                        G_f3_27_q(i,j) = new_val_f3_27(3);  
                    elseif G_f3_27(i,j) >= edges(4) && G_f3_27(i,j) < edges(5)  
                        G_f3_27_q(i,j) = new_val_f3_27(4);   
                    elseif G_f3_27(i,j) >= edges(5) && G_f3_27(i,j) < edges(6)  
                        G_f3_27_q(i,j) = new_val_f3_27(5);     
                    elseif G_f3_27(i,j) >= edges(6) && G_f3_27(i,j) < edges(7)  
                        G_f3_27_q(i,j) = new_val_f3_27(6);  
                    elseif G_f3_27(i,j) >= edges(7) && G_f3_27(i,j) < edges(8)  
                        G_f3_27_q(i,j) = new_val_f3_27(7);   
                    elseif G_f3_27(i,j) >= edges(8) && G_f3_27(i,j) < edges(9)  
                        G_f3_27_q(i,j) = new_val_f3_27(8);     
                    elseif G_f3_27(i,j) >= edges(9) && G_f3_27(i,j) < edges(10)  
                        G_f3_27_q(i,j) = new_val_f3_27(9);  
                    elseif G_f3_27(i,j) >= edges(10) && G_f3_27(i,j) < edges(11)  
                        G_f3_27_q(i,j) = new_val_f3_27(10);  
                    elseif G_f3_27(i,j) >= edges(11) && G_f3_27(i,j) < edges(12)  
                        G_f3_27_q(i,j) = new_val_f3_27(11);  
                    elseif G_f3_27(i,j) >= edges(12) && G_f3_27(i,j) < edges(13)  
                        G_f3_27_q(i,j) = new_val_f3_27(12);  
                    elseif G_f3_27(i,j) >= edges(13) && G_f3_27(i,j) < edges(14)   
                         G_f3_27_q(i,j) = new_val_f3_27(13);   
                    elseif G_f3_27(i,j) >= edges(14) && G_f3_27(i,j) < edges(15)  
                        G_f3_27_q(i,j) = new_val_f3_27(14);  
                    elseif G_f3_27(i,j) >= edges(15) && G_f3_27(i,j) < edges(16)  
                        G_f3_27_q(i,j) = new_val_f3_27(15);  
                    elseif G_f3_27(i,j) >= edges(16) && G_f3_27(i,j) <= edges(17)   
                         G_f3_27_q(i,j) = new_val_f3_27(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_28,1) 
                for j = 1:size(G_f3_28,2) 
                    if G_f3_28(i,j) < edges(1)  
                        G_f3_28_q(i,j) = new_val_f3_28(1);  
                    elseif G_f3_28(i,j) >= edges(1) && G_f3_28(i,j) < edges(2)  
                        G_f3_28_q(i,j) = new_val_f3_28(1);  
                    elseif G_f3_28(i,j) >= edges(2) && G_f3_28(i,j) < edges(3)  
                        G_f3_28_q(i,j) = new_val_f3_28(2);    
                    elseif G_f3_28(i,j) >= edges(3) && G_f3_28(i,j) < edges(4)  
                        G_f3_28_q(i,j) = new_val_f3_28(3);  
                    elseif G_f3_28(i,j) >= edges(4) && G_f3_28(i,j) < edges(5)  
                        G_f3_28_q(i,j) = new_val_f3_28(4);   
                    elseif G_f3_28(i,j) >= edges(5) && G_f3_28(i,j) < edges(6)  
                        G_f3_28_q(i,j) = new_val_f3_28(5);     
                    elseif G_f3_28(i,j) >= edges(6) && G_f3_28(i,j) < edges(7)  
                        G_f3_28_q(i,j) = new_val_f3_28(6);  
                    elseif G_f3_28(i,j) >= edges(7) && G_f3_28(i,j) < edges(8)  
                        G_f3_28_q(i,j) = new_val_f3_28(7);   
                    elseif G_f3_28(i,j) >= edges(8) && G_f3_28(i,j) < edges(9)  
                        G_f3_28_q(i,j) = new_val_f3_28(8);     
                    elseif G_f3_28(i,j) >= edges(9) && G_f3_28(i,j) < edges(10)  
                        G_f3_28_q(i,j) = new_val_f3_28(9);  
                    elseif G_f3_28(i,j) >= edges(10) && G_f3_28(i,j) < edges(11)  
                        G_f3_28_q(i,j) = new_val_f3_28(10);  
                    elseif G_f3_28(i,j) >= edges(11) && G_f3_28(i,j) < edges(12)  
                        G_f3_28_q(i,j) = new_val_f3_28(11);  
                    elseif G_f3_28(i,j) >= edges(12) && G_f3_28(i,j) < edges(13)  
                        G_f3_28_q(i,j) = new_val_f3_28(12);  
                    elseif G_f3_28(i,j) >= edges(13) && G_f3_28(i,j) < edges(14)   
                         G_f3_28_q(i,j) = new_val_f3_28(13);   
                    elseif G_f3_28(i,j) >= edges(14) && G_f3_28(i,j) < edges(15)  
                        G_f3_28_q(i,j) = new_val_f3_28(14);  
                    elseif G_f3_28(i,j) >= edges(15) && G_f3_28(i,j) < edges(16)  
                        G_f3_28_q(i,j) = new_val_f3_28(15);  
                    elseif G_f3_28(i,j) >= edges(16) && G_f3_28(i,j) <= edges(17)   
                         G_f3_28_q(i,j) = new_val_f3_28(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_29,1) 
                for j = 1:size(G_f3_29,2) 
                    if G_f3_29(i,j) < edges(1)  
                        G_f3_29_q(i,j) = new_val_f3_29(1);  
                    elseif G_f3_29(i,j) >= edges(1) && G_f3_29(i,j) < edges(2)  
                        G_f3_29_q(i,j) = new_val_f3_29(1);  
                    elseif G_f3_29(i,j) >= edges(2) && G_f3_29(i,j) < edges(3)  
                        G_f3_29_q(i,j) = new_val_f3_29(2);    
                    elseif G_f3_29(i,j) >= edges(3) && G_f3_29(i,j) < edges(4)  
                        G_f3_29_q(i,j) = new_val_f3_29(3);  
                    elseif G_f3_29(i,j) >= edges(4) && G_f3_29(i,j) < edges(5)  
                        G_f3_29_q(i,j) = new_val_f3_29(4);   
                    elseif G_f3_29(i,j) >= edges(5) && G_f3_29(i,j) < edges(6)  
                        G_f3_29_q(i,j) = new_val_f3_29(5);     
                    elseif G_f3_29(i,j) >= edges(6) && G_f3_29(i,j) < edges(7)  
                        G_f3_29_q(i,j) = new_val_f3_29(6);  
                    elseif G_f3_29(i,j) >= edges(7) && G_f3_29(i,j) < edges(8)  
                        G_f3_29_q(i,j) = new_val_f3_29(7);   
                    elseif G_f3_29(i,j) >= edges(8) && G_f3_29(i,j) < edges(9)  
                        G_f3_29_q(i,j) = new_val_f3_29(8);     
                    elseif G_f3_29(i,j) >= edges(9) && G_f3_29(i,j) < edges(10)  
                        G_f3_29_q(i,j) = new_val_f3_29(9);  
                    elseif G_f3_29(i,j) >= edges(10) && G_f3_29(i,j) < edges(11)  
                        G_f3_29_q(i,j) = new_val_f3_29(10);  
                    elseif G_f3_29(i,j) >= edges(11) && G_f3_29(i,j) < edges(12)  
                        G_f3_29_q(i,j) = new_val_f3_29(11);  
                    elseif G_f3_29(i,j) >= edges(12) && G_f3_29(i,j) < edges(13)  
                        G_f3_29_q(i,j) = new_val_f3_29(12);  
                    elseif G_f3_29(i,j) >= edges(13) && G_f3_29(i,j) < edges(14)   
                         G_f3_29_q(i,j) = new_val_f3_29(13);   
                    elseif G_f3_29(i,j) >= edges(14) && G_f3_29(i,j) < edges(15)  
                        G_f3_29_q(i,j) = new_val_f3_29(14);  
                    elseif G_f3_29(i,j) >= edges(15) && G_f3_29(i,j) < edges(16)  
                        G_f3_29_q(i,j) = new_val_f3_29(15);  
                    elseif G_f3_29(i,j) >= edges(16) && G_f3_29(i,j) <= edges(17)   
                         G_f3_29_q(i,j) = new_val_f3_29(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_30,1) 
                for j = 1:size(G_f3_30,2) 
                    if G_f3_30(i,j) < edges(1)  
                        G_f3_30_q(i,j) = new_val_f3_30(1);  
                    elseif G_f3_30(i,j) >= edges(1) && G_f3_30(i,j) < edges(2)  
                        G_f3_30_q(i,j) = new_val_f3_30(1);  
                    elseif G_f3_30(i,j) >= edges(2) && G_f3_30(i,j) < edges(3)  
                        G_f3_30_q(i,j) = new_val_f3_30(2);    
                    elseif G_f3_30(i,j) >= edges(3) && G_f3_30(i,j) < edges(4)  
                        G_f3_30_q(i,j) = new_val_f3_30(3);  
                    elseif G_f3_30(i,j) >= edges(4) && G_f3_30(i,j) < edges(5)  
                        G_f3_30_q(i,j) = new_val_f3_30(4);   
                    elseif G_f3_30(i,j) >= edges(5) && G_f3_30(i,j) < edges(6)  
                        G_f3_30_q(i,j) = new_val_f3_30(5);     
                    elseif G_f3_30(i,j) >= edges(6) && G_f3_30(i,j) < edges(7)  
                        G_f3_30_q(i,j) = new_val_f3_30(6);  
                    elseif G_f3_30(i,j) >= edges(7) && G_f3_30(i,j) < edges(8)  
                        G_f3_30_q(i,j) = new_val_f3_30(7);   
                    elseif G_f3_30(i,j) >= edges(8) && G_f3_30(i,j) < edges(9)  
                        G_f3_30_q(i,j) = new_val_f3_30(8);     
                    elseif G_f3_30(i,j) >= edges(9) && G_f3_30(i,j) < edges(10)  
                        G_f3_30_q(i,j) = new_val_f3_30(9);  
                    elseif G_f3_30(i,j) >= edges(10) && G_f3_30(i,j) < edges(11)  
                        G_f3_30_q(i,j) = new_val_f3_30(10);  
                    elseif G_f3_30(i,j) >= edges(11) && G_f3_30(i,j) < edges(12)  
                        G_f3_30_q(i,j) = new_val_f3_30(11);  
                    elseif G_f3_30(i,j) >= edges(12) && G_f3_30(i,j) < edges(13)  
                        G_f3_30_q(i,j) = new_val_f3_30(12);  
                    elseif G_f3_30(i,j) >= edges(13) && G_f3_30(i,j) < edges(14)   
                         G_f3_30_q(i,j) = new_val_f3_30(13);   
                    elseif G_f3_30(i,j) >= edges(14) && G_f3_30(i,j) < edges(15)  
                        G_f3_30_q(i,j) = new_val_f3_30(14);  
                    elseif G_f3_30(i,j) >= edges(15) && G_f3_30(i,j) < edges(16)  
                        G_f3_30_q(i,j) = new_val_f3_30(15);  
                    elseif G_f3_30(i,j) >= edges(16) && G_f3_30(i,j) <= edges(17)   
                         G_f3_30_q(i,j) = new_val_f3_30(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_31,1) 
                for j = 1:size(G_f3_31,2) 
                    if G_f3_31(i,j) < edges(1)  
                        G_f3_31_q(i,j) = new_val_f3_31(1);  
                    elseif G_f3_31(i,j) >= edges(1) && G_f3_31(i,j) < edges(2)  
                        G_f3_31_q(i,j) = new_val_f3_31(1);  
                    elseif G_f3_31(i,j) >= edges(2) && G_f3_31(i,j) < edges(3)  
                        G_f3_31_q(i,j) = new_val_f3_31(2);    
                    elseif G_f3_31(i,j) >= edges(3) && G_f3_31(i,j) < edges(4)  
                        G_f3_31_q(i,j) = new_val_f3_31(3);  
                    elseif G_f3_31(i,j) >= edges(4) && G_f3_31(i,j) < edges(5)  
                        G_f3_31_q(i,j) = new_val_f3_31(4);   
                    elseif G_f3_31(i,j) >= edges(5) && G_f3_31(i,j) < edges(6)  
                        G_f3_31_q(i,j) = new_val_f3_31(5);     
                    elseif G_f3_31(i,j) >= edges(6) && G_f3_31(i,j) < edges(7)  
                        G_f3_31_q(i,j) = new_val_f3_31(6);  
                    elseif G_f3_31(i,j) >= edges(7) && G_f3_31(i,j) < edges(8)  
                        G_f3_31_q(i,j) = new_val_f3_31(7);   
                    elseif G_f3_31(i,j) >= edges(8) && G_f3_31(i,j) < edges(9)  
                        G_f3_31_q(i,j) = new_val_f3_31(8);     
                    elseif G_f3_31(i,j) >= edges(9) && G_f3_31(i,j) < edges(10)  
                        G_f3_31_q(i,j) = new_val_f3_31(9);  
                    elseif G_f3_31(i,j) >= edges(10) && G_f3_31(i,j) < edges(11)  
                        G_f3_31_q(i,j) = new_val_f3_31(10);  
                    elseif G_f3_31(i,j) >= edges(11) && G_f3_31(i,j) < edges(12)  
                        G_f3_31_q(i,j) = new_val_f3_31(11);  
                    elseif G_f3_31(i,j) >= edges(12) && G_f3_31(i,j) < edges(13)  
                        G_f3_31_q(i,j) = new_val_f3_31(12);  
                    elseif G_f3_31(i,j) >= edges(13) && G_f3_31(i,j) < edges(14)   
                         G_f3_31_q(i,j) = new_val_f3_31(13);   
                    elseif G_f3_31(i,j) >= edges(14) && G_f3_31(i,j) < edges(15)  
                        G_f3_31_q(i,j) = new_val_f3_31(14);  
                    elseif G_f3_31(i,j) >= edges(15) && G_f3_31(i,j) < edges(16)  
                        G_f3_31_q(i,j) = new_val_f3_31(15);  
                    elseif G_f3_31(i,j) >= edges(16) && G_f3_31(i,j) <= edges(17)   
                         G_f3_31_q(i,j) = new_val_f3_31(16);    
                    end 
                end 
            end  


            for i = 1:size(G_f3_32,1) 
                for j = 1:size(G_f3_32,2) 
                    if G_f3_32(i,j) < edges(1)  
                        G_f3_32_q(i,j) = new_val_f3_32(1);  
                    elseif G_f3_32(i,j) >= edges(1) && G_f3_32(i,j) < edges(2)  
                        G_f3_32_q(i,j) = new_val_f3_32(1);  
                    elseif G_f3_32(i,j) >= edges(2) && G_f3_32(i,j) < edges(3)  
                        G_f3_32_q(i,j) = new_val_f3_32(2);    
                    elseif G_f3_32(i,j) >= edges(3) && G_f3_32(i,j) < edges(4)  
                        G_f3_32_q(i,j) = new_val_f3_32(3);  
                    elseif G_f3_32(i,j) >= edges(4) && G_f3_32(i,j) < edges(5)  
                        G_f3_32_q(i,j) = new_val_f3_32(4);   
                    elseif G_f3_32(i,j) >= edges(5) && G_f3_32(i,j) < edges(6)  
                        G_f3_32_q(i,j) = new_val_f3_32(5);     
                    elseif G_f3_32(i,j) >= edges(6) && G_f3_32(i,j) < edges(7)  
                        G_f3_32_q(i,j) = new_val_f3_32(6);  
                    elseif G_f3_32(i,j) >= edges(7) && G_f3_32(i,j) < edges(8)  
                        G_f3_32_q(i,j) = new_val_f3_32(7);   
                    elseif G_f3_32(i,j) >= edges(8) && G_f3_32(i,j) < edges(9)  
                        G_f3_32_q(i,j) = new_val_f3_32(8);     
                    elseif G_f3_32(i,j) >= edges(9) && G_f3_32(i,j) < edges(10)  
                        G_f3_32_q(i,j) = new_val_f3_32(9);  
                    elseif G_f3_32(i,j) >= edges(10) && G_f3_32(i,j) < edges(11)  
                        G_f3_32_q(i,j) = new_val_f3_32(10);  
                    elseif G_f3_32(i,j) >= edges(11) && G_f3_32(i,j) < edges(12)  
                        G_f3_32_q(i,j) = new_val_f3_32(11);  
                    elseif G_f3_32(i,j) >= edges(12) && G_f3_32(i,j) < edges(13)  
                        G_f3_32_q(i,j) = new_val_f3_32(12);  
                    elseif G_f3_32(i,j) >= edges(13) && G_f3_32(i,j) < edges(14)   
                         G_f3_32_q(i,j) = new_val_f3_32(13);   
                    elseif G_f3_32(i,j) >= edges(14) && G_f3_32(i,j) < edges(15)  
                        G_f3_32_q(i,j) = new_val_f3_32(14);  
                    elseif G_f3_32(i,j) >= edges(15) && G_f3_32(i,j) < edges(16)  
                        G_f3_32_q(i,j) = new_val_f3_32(15);  
                    elseif G_f3_32(i,j) >= edges(16) && G_f3_32(i,j) <= edges(17)   
                         G_f3_32_q(i,j) = new_val_f3_32(16);    
                    end 
                end 
            end  
        
        out_1_3_cond = im_cond_3_1*G_f3_1_q; 
        out_2_3_cond = im_cond_3_2*G_f3_2_q; 
        out_3_3_cond = im_cond_3_3*G_f3_3_q; 
        out_4_3_cond = im_cond_3_4*G_f3_4_q; 
        out_5_3_cond = im_cond_3_5*G_f3_5_q; 
        out_6_3_cond = im_cond_3_6*G_f3_6_q; 
        out_7_3_cond = im_cond_3_7*G_f3_7_q; 
        out_8_3_cond = im_cond_3_8*G_f3_8_q; 
        out_9_3_cond = im_cond_3_9*G_f3_9_q; 
        out_10_3_cond = im_cond_3_10*G_f3_10_q; 
        out_11_3_cond = im_cond_3_11*G_f3_11_q; 
        out_12_3_cond = im_cond_3_12*G_f3_12_q; 
        out_13_3_cond = im_cond_3_13*G_f3_13_q; 
        out_14_3_cond = im_cond_3_14*G_f3_14_q; 
        out_15_3_cond = im_cond_3_15*G_f3_15_q; 
        out_16_3_cond = im_cond_3_16*G_f3_16_q; 
        out_17_3_cond = im_cond_3_17*G_f3_17_q; 
        out_18_3_cond = im_cond_3_18*G_f3_18_q; 
        out_19_3_cond = im_cond_3_19*G_f3_19_q; 
        out_20_3_cond = im_cond_3_20*G_f3_20_q; 
        out_21_3_cond = im_cond_3_21*G_f3_21_q; 
        out_22_3_cond = im_cond_3_22*G_f3_22_q; 
        out_23_3_cond = im_cond_3_23*G_f3_23_q; 
        out_24_3_cond = im_cond_3_24*G_f3_24_q; 
        out_25_3_cond = im_cond_3_25*G_f3_25_q; 
        out_26_3_cond = im_cond_3_26*G_f3_26_q; 
        out_27_3_cond = im_cond_3_27*G_f3_27_q; 
        out_28_3_cond = im_cond_3_28*G_f3_28_q; 
        out_29_3_cond = im_cond_3_29*G_f3_29_q; 
        out_30_3_cond = im_cond_3_30*G_f3_30_q; 
        out_31_3_cond = im_cond_3_31*G_f3_31_q; 
        out_32_3_cond = im_cond_3_32*G_f3_32_q; 

        output_third_layer_with_shifting_cond = out_1_3_cond + out_2_3_cond +...
            out_3_3_cond + out_4_3_cond + out_5_3_cond + out_6_3_cond + out_7_3_cond +...
            out_8_3_cond + out_9_3_cond + out_10_3_cond + out_11_3_cond + out_12_3_cond +...
            out_13_3_cond + out_14_3_cond + out_15_3_cond + out_16_3_cond + out_17_3_cond +...
            out_18_3_cond + out_19_3_cond + out_20_3_cond + out_21_3_cond + out_22_3_cond +...
            out_23_3_cond + out_24_3_cond + out_25_3_cond + out_26_3_cond + out_27_3_cond +...
            out_28_3_cond + out_29_3_cond + out_30_3_cond + out_31_3_cond + out_32_3_cond ;


        shifting_matrix_cond3 = output_third_layer_with_shifting_cond(:,65:end);
        output_third_layer_with_shifting_cond = output_third_layer_with_shifting_cond(:,1:64);

        % Each col is an output filter (its flattening happened row-by-row)
        output_third_layer_with_shifting_cond = output_third_layer_with_shifting_cond - shifting_matrix_cond3;

        % Re-shape it to pass it through pooling and relu
        output_3_cond = permute(reshape(output_third_layer_with_shifting_cond,[8,8,64]),[2 1 3]);


        % ReLU 2
        for i =1: size(output_3_shifting,1)
            for j = 1: size(output_3_shifting,2)
                for k = 1: size(output_3_shifting,3)
                    if output_3_shifting(i,j,k) < 0
                        output_3_shifting(i,j,k) = 0;
                    end
                end
            end
        end

        for i =1: size(output_3_cond,1)
            for j = 1: size(output_3_cond,2)
                for k = 1: size(output_3_cond,3)
                    if output_3_cond(i,j,k) < 0
                        output_3_cond(i,j,k) = 0;
                    end
                end
            end
        end

        % Average Pooling
        PoolMap3_shifting = zeros(size(pool3,1),size(pool3,2), size(pool3,3));
        PoolMap3_shifting_cond = zeros(size(pool3,1),size(pool3,2), size(pool3,3));

        row1 = 1;
        col1 = 3;
        row2 = 1;
        col2 = 3;
        k = 3; % kernel size
        for j = 1 : numOfFilters3 %across all filters
            k1 = 1;
            for s1 = 0:2:(size(output_3_shifting,1)-2) %controls the shifting of the row and col per col
                k2 = 1;
                for s2 = 0:2:(size(output_3_shifting,1)-2) %controls the shifting of the col
                    if (s1 == (size(output_3_shifting,1)-2)) && (s2 == (size(output_3_shifting,1)-2))
                        out1 = sum(sum(output_3_shifting(row1+s1:col1+s1-1,row2+s2:col2+s2-1,j)))/((k-1)*(k-1));
                    elseif s2 == (size(output_3_shifting,1)-2)
                        out1 = sum(sum(output_3_shifting(row1+s1:col1+s1,row2+s2:col2+s2-1,j)))/((k-1)*k); 
                    elseif s1 == (size(output_3_shifting,1)-2)
                        out1 = sum(sum(output_3_shifting(row1+s1:col1+s1-1,row2+s2:col2+s2,j)))/((k-1)*k); 
                    else 
                        out1 = sum(sum(output_3_shifting(row1+s1:col1+s1,row2+s2:col2+s2,j)))/(k*k);  
                    end    
                    PoolMap3_shifting(k1,k2,j) = out1;
                    k2 = k2 + 1;

                end
                k1 = k1 + 1;
            end  
        end

        for j = 1 : numOfFilters3 %across all filters
            k1 = 1;
            for s1 = 0:2:(size(output_3_cond,1)-2) %controls the shifting of the row and col per col
                k2 = 1;
                for s2 = 0:2:(size(output_3_cond,1)-2) %controls the shifting of the col
                    if (s1 == (size(output_3_cond,1)-2)) && (s2 == (size(output_3_cond,1)-2))
                        out1 = sum(sum(output_3_cond(row1+s1:col1+s1-1,row2+s2:col2+s2-1,j)))/((k-1)*(k-1));
                    elseif s2 == (size(output_3_cond,1)-2)
                        out1 = sum(sum(output_3_cond(row1+s1:col1+s1,row2+s2:col2+s2-1,j)))/((k-1)*k); 
                    elseif s1 == (size(output_3_cond,1)-2)
                        out1 = sum(sum(output_3_cond(row1+s1:col1+s1-1,row2+s2:col2+s2,j)))/((k-1)*k); 
                    else 
                        out1 = sum(sum(output_3_cond(row1+s1:col1+s1,row2+s2:col2+s2,j)))/(k*k);  
                    end    
                    PoolMap3_shifting_cond(k1,k2,j) = out1;
                    k2 = k2 + 1;

                end
                k1 = k1 + 1;
            end  
        end

        PoolMap3_shifting_cond = PoolMap3_shifting_cond * 26;
        
         %% First FC Layer
        flatten_PoolMap3_shifting = zeros(1, size(PoolMap3_shifting,1)*size(PoolMap3_shifting,2)*size(PoolMap3_shifting,3));
        flatten_PoolMap3_shifting_cond = zeros(1, size(PoolMap3_shifting,1)*size(PoolMap3_shifting,2)*size(PoolMap3_shifting,3));

        dim = numel(PoolMap3_shifting(:,:,1));

        k = 0;
        for j = 1: numOfFilters3
            pool_flatten(:,:) = PoolMap3_shifting(:,:,j);
            flatten_PoolMap3_shifting(1,1+k:dim+k) = horzcat(reshape(pool_flatten, 1,dim));
            k = k + dim;
        end

        k = 0;
        for j = 1: numOfFilters3
            pool_flatten(:,:) = PoolMap3_shifting_cond(:,:,j);
            flatten_PoolMap3_shifting_cond(1,1+k:dim+k) = horzcat(reshape(pool_flatten, 1,dim));
            k = k + dim;
        end


        FC_weights_1 = [FC_weights_1 ; FC_bias_1'];
        shift_FC1 = min(min(FC_weights_1));
        FC_weights_1 = FC_weights_1 - shift_FC1;
        FC_weights_1 = [FC_weights_1, abs(shift_FC1.*ones(size(FC_weights_1,1),1))];

        minValue = min(FC_weights_1(:));
        maxValue = max(FC_weights_1(:));

        a_h = (Gon - Goff)/(maxValue - minValue);
        b_h = Gon - a_h *(maxValue);
        G_FC_weights_1 = a_h .* FC_weights_1 + b_h;

        % pad with 1 for the bias
        flatten_PoolMap3_shifting = [flatten_PoolMap3_shifting 1];
        flatten_PoolMap3_shifting_cond = [flatten_PoolMap3_shifting_cond 1];

        output_4_shifting =  flatten_PoolMap3_shifting*FC_weights_1;  
        
        new_val_FC1 = cell(1,1);

        for i = 1: length(edges)-1
            range_mean = [edges(i) edges(i+1)];
            
            pd = makedist('normal','mu', mean(range_mean), 'sigma', mean(range_mean) + (variation*mean(range_mean)));
            upper = mean(range_mean) + (variation*mean(range_mean));
            lower = mean(range_mean) - (variation*mean(range_mean));
            t = truncate(pd, lower,upper);

            new_val_FC1{i} = random(t,1,1);  
        end
        
        new_val_FC1 = cell2mat(new_val_FC1); 
      
       G_FC_weights_1_q = zeros(size(G_FC_weights_1,1),size(G_FC_weights_1,2));
 
            
            
                for i = 1:size(G_FC_weights_1,1)
                    for j = 1:size(G_FC_weights_1,2)
                        if G_FC_weights_1(i,j) < edges(1) 
                            G_FC_weights_1_q(i,j) = new_val_FC1(1);

                        elseif G_FC_weights_1(i,j) >= edges(1) && G_FC_weights_1(i,j) < edges(2)
                            G_FC_weights_1_q(i,j) = new_val_FC1(1);


                        elseif G_FC_weights_1(i,j) >= edges(2) && G_FC_weights_1(i,j) < edges(3)
                            G_FC_weights_1_q(i,j) = new_val_FC1(2);   

                        elseif G_FC_weights_1(i,j) >= edges(3) && G_FC_weights_1(i,j) < edges(4)
                            G_FC_weights_1_q(i,j) = new_val_FC1(3);

                        elseif G_FC_weights_1(i,j) >= edges(4) && G_FC_weights_1(i,j) < edges(5)
                            G_FC_weights_1_q(i,j) = new_val_FC1(4); 

                        elseif G_FC_weights_1(i,j) >= edges(5) && G_FC_weights_1(i,j) < edges(6)
                            G_FC_weights_1_q(i,j) = new_val_FC1(5);          

                        elseif G_FC_weights_1(i,j) >= edges(6) && G_FC_weights_1(i,j) < edges(7)
                            G_FC_weights_1_q(i,j) = new_val_FC1(6);

                        elseif G_FC_weights_1(i,j) >= edges(7) && G_FC_weights_1(i,j) < edges(8)
                            G_FC_weights_1_q(i,j) = new_val_FC1(7); 

                        elseif G_FC_weights_1(i,j) >= edges(8) && G_FC_weights_1(i,j) < edges(9)
                            G_FC_weights_1_q(i,j) = new_val_FC1(8);          

                        elseif G_FC_weights_1(i,j) >= edges(9) && G_FC_weights_1(i,j) < edges(10)
                            G_FC_weights_1_q(i,j) = new_val_FC1(9);

                        elseif G_FC_weights_1(i,j) >= edges(10) && G_FC_weights_1(i,j) < edges(11)
                            G_FC_weights_1_q(i,j) = new_val_FC1(10);

                        elseif G_FC_weights_1(i,j) >= edges(11) && G_FC_weights_1(i,j) < edges(12)
                            G_FC_weights_1_q(i,j) = new_val_FC1(11);

                        elseif G_FC_weights_1(i,j) >= edges(12) && G_FC_weights_1(i,j) < edges(13)
                            G_FC_weights_1_q(i,j) = new_val_FC1(12);

                        elseif G_FC_weights_1(i,j) >= edges(13) && G_FC_weights_1(i,j) < edges(14) 
                             G_FC_weights_1_q(i,j) = new_val_FC1(13);

                        elseif G_FC_weights_1(i,j) >= edges(14) && G_FC_weights_1(i,j) < edges(15) 
                            G_FC_weights_1_q(i,j) = new_val_FC1(14);

                        elseif G_FC_weights_1(i,j) >= edges(15) && G_FC_weights_1(i,j) < edges(16)
                            G_FC_weights_1_q(i,j) = new_val_FC1(15);

                        elseif G_FC_weights_1(i,j) >= edges(16) && G_FC_weights_1(i,j) <= edges(17) 
                             G_FC_weights_1_q(i,j) = new_val_FC1(16);            
                        end
                    end
                end  

        
        output_4_shifting_cond =  flatten_PoolMap3_shifting_cond*G_FC_weights_1_q;   

        output_4_shifting = output_4_shifting - output_4_shifting(end);
        output_4_shifting = output_4_shifting(1,1:end-1);

        output_4_shifting_cond = output_4_shifting_cond - output_4_shifting_cond(end);
        output_4_shifting_cond = output_4_shifting_cond(1,1:end-1);

        output_4_shifting_cond = output_4_shifting_cond * 34; 
        
         %% Second FC Layer
   
        FC_weights_2 = [FC_weights_2 ; FC_bias_2'];
        shift_FC2 = min(min(FC_weights_2));
        FC_weights_2 = FC_weights_2 - shift_FC2;
        FC_weights_2 = [FC_weights_2, abs(shift_FC2.*ones(size(FC_weights_2,1),1))];

        minValue = min(FC_weights_2(:));
        maxValue = max(FC_weights_2(:));

        a_h = (Gon - Goff)/(maxValue - minValue);
        b_h = Gon - a_h *(maxValue);
        G_FC_weights_2 = a_h .* FC_weights_2 + b_h;


        % pad with 1 for the bias
        output_4_shifting = [output_4_shifting 1];
        output_4_shifting_cond = [output_4_shifting_cond 1];

        output_5_shifting =  output_4_shifting*FC_weights_2;    
        output_5_shifting = output_5_shifting - output_5_shifting(end);
        output_5_shifting = output_5_shifting(1,1:end-1);

              
        new_val_FC2 = cell(1,1);        
        for i = 1: length(edges)-1
            range_mean = [edges(i) edges(i+1)];
            
            pd = makedist('normal','mu', mean(range_mean), 'sigma', mean(range_mean) + (variation*mean(range_mean)));
            upper = mean(range_mean) + (variation*mean(range_mean));
            lower = mean(range_mean) - (variation*mean(range_mean));
            t = truncate(pd, lower,upper);
            new_val_FC2{i} = random(t,1,1);
              
        end
        new_val_FC2 = cell2mat(new_val_FC2); 
                

        G_FC_weights_2_q = zeros(size(G_FC_weights_2,1),size(G_FC_weights_2,2));
                    
                    
                  for i = 1:size(G_FC_weights_2,1)
                    for j = 1:size(G_FC_weights_2,2)
                        if G_FC_weights_2(i,j) < edges(1) 
                            G_FC_weights_2_q(i,j) = new_val_FC2(1);

                        elseif G_FC_weights_2(i,j) >= edges(1) && G_FC_weights_2(i,j) < edges(2)
                            G_FC_weights_2_q(i,j) = new_val_FC2(1);


                        elseif G_FC_weights_2(i,j) >= edges(2) && G_FC_weights_2(i,j) < edges(3)
                            G_FC_weights_2_q(i,j) = new_val_FC2(2);   

                        elseif G_FC_weights_2(i,j) >= edges(3) && G_FC_weights_2(i,j) < edges(4)
                            G_FC_weights_2_q(i,j) = new_val_FC2(3);

                        elseif G_FC_weights_2(i,j) >= edges(4) && G_FC_weights_2(i,j) < edges(5)
                            G_FC_weights_2_q(i,j) = new_val_FC2(4); 

                        elseif G_FC_weights_2(i,j) >= edges(5) && G_FC_weights_2(i,j) < edges(6)
                            G_FC_weights_2_q(i,j) = new_val_FC2(5);          

                        elseif G_FC_weights_2(i,j) >= edges(6) && G_FC_weights_2(i,j) < edges(7)
                            G_FC_weights_2_q(i,j) = new_val_FC2(6);

                        elseif G_FC_weights_2(i,j) >= edges(7) && G_FC_weights_2(i,j) < edges(8)
                            G_FC_weights_2_q(i,j) = new_val_FC2(7); 

                        elseif G_FC_weights_2(i,j) >= edges(8) && G_FC_weights_2(i,j) < edges(9)
                            G_FC_weights_2_q(i,j) = new_val_FC2(8);          

                        elseif G_FC_weights_2(i,j) >= edges(9) && G_FC_weights_2(i,j) < edges(10)
                            G_FC_weights_2_q(i,j) = new_val_FC2(9);

                        elseif G_FC_weights_2(i,j) >= edges(10) && G_FC_weights_2(i,j) < edges(11)
                            G_FC_weights_2_q(i,j) = new_val_FC2(10);

                        elseif G_FC_weights_2(i,j) >= edges(11) && G_FC_weights_2(i,j) < edges(12)
                            G_FC_weights_2_q(i,j) = new_val_FC2(11);

                        elseif G_FC_weights_2(i,j) >= edges(12) && G_FC_weights_2(i,j) < edges(13)
                            G_FC_weights_2_q(i,j) = new_val_FC2(12);

                        elseif G_FC_weights_2(i,j) >= edges(13) && G_FC_weights_2(i,j) < edges(14) 
                             G_FC_weights_2_q(i,j) = new_val_FC2(13);

                        elseif G_FC_weights_2(i,j) >= edges(14) && G_FC_weights_2(i,j) < edges(15) 
                            G_FC_weights_2_q(i,j) = new_val_FC2(14);

                        elseif G_FC_weights_2(i,j) >= edges(15) && G_FC_weights_2(i,j) < edges(16)
                            G_FC_weights_2_q(i,j) = new_val_FC2(15);

                        elseif G_FC_weights_2(i,j) >= edges(16) && G_FC_weights_2(i,j) <= edges(17) 
                             G_FC_weights_2_q(i,j) = new_val_FC2(16);            
                        end
                    end
                  end  

        output_5_shifting_cond =  output_4_shifting_cond*G_FC_weights_2_q;    
        output_5_shifting_cond = output_5_shifting_cond - output_5_shifting_cond(end);
        output_5_shifting_cond = output_5_shifting_cond(1,1:end-1);

        output_5_shifting_cond = output_5_shifting_cond * 43;


        %% Softmax Output Layer
       
        Classification_output_shifting = softmax(output_5_shifting');
        [score, index] = max(Classification_output_shifting);
        winning_class = labels(index); 

        Classification_output_shifting_cond = softmax(output_5_shifting_cond');
        [score_cond, index_cond] = max(Classification_output_shifting_cond);
        winning_class_cond = labels(index_cond); 
        
        
        if strcmp(winning_class_cond,labels(m)) == 1 
            total_accuracy_cond = total_accuracy_cond + 1;
   
        end 

        
    end 
end 

        
        
        