
%% the process of generating the benchmark random network

function adj = rand_network( zout )

numVex = 128;

numCluster = 4;

numNode = 32;

degree = 16;

% zout = 1;

zin = degree - zout;

numIntraLink = (zin*numNode*numCluster)/2;   % intra：内部链接

numInterLink = (zout*numNode*numCluster)/2;  % inter：外部链接

maxNumIntraLink = numCluster * ( numNode * (numNode-1) ) / 2;   % 假设有4个社区，任意两个社区之间没有连接

maxNumInterLink = ( numVex * (numVex-1) ) / 2;   % 每个节点都有外部链接

if numIntraLink>maxNumIntraLink
    fprintf('Error! There cannot be so many intralinks!\n');
    exit
end

if numInterLink>maxNumInterLink
    fprintf('Error! There cannot be so many inter links!\n');
    exit
end

adj = zeros( numVex, numVex );

z_in_arr = zeros(1,numVex);
z_out_arr = zeros(1,numVex);

community = [];
for i = 1:numCluster
    for j = 1:numNode
        community{i}(j) = (i-1)*32+j;
    end
end

for i = 1:numCluster
    
    com_id_i = community{i};
    
    for j = 1:numNode
        
        cur_node_id = com_id_i(j);

        temp_com_id_i = com_id_i;
        temp_com_id_i(j) = [];

        k = 1;
        while k <= length(temp_com_id_i)

            if adj( cur_node_id, temp_com_id_i(k) ) == 1
                temp_com_id_i(k) = [];
            else
                k = k + 1;
            end
        end
        
        exist_intra_link = (numNode-1) - length(temp_com_id_i);
        
        if z_in_arr(cur_node_id) ~= exist_intra_link
            fprintf('Error has occured: z_in_arr(cur_node_id) ~= exist_intra_link!!!\n');
            exit;
        end
        
        zin_left = zin - z_in_arr(cur_node_id);
        
        flag = 0;

        for k = 1:zin_left

            v = 1;
            while v <= length(temp_com_id_i)
                if z_in_arr(temp_com_id_i(v)) == zin
                    temp_com_id_i(v) = [];
                else
                    v = v + 1;
                end
            end
            
            if isempty(temp_com_id_i)
                flag = 1;
                break
            end
            
            rand_index = floor(rand*length(temp_com_id_i)) + 1;
            
            conn_node_id = temp_com_id_i( rand_index );
            
            if adj( cur_node_id, conn_node_id ) == 1
                fprintf('Error: %d and %d have alreadly connected!!!\n',cur_node_id,conn_node_id);
                exit;
            end
            
            adj( cur_node_id, conn_node_id ) = 1;
            adj( conn_node_id, cur_node_id ) = 1;
            
            z_in_arr(cur_node_id) = z_in_arr(cur_node_id) + 1;
            z_in_arr(conn_node_id) = z_in_arr(conn_node_id) + 1;
            
            temp_com_id_i( rand_index ) = [];
            
        end
        
        if flag==1
            
            remain_num = zin - z_in_arr(cur_node_id);
            
            temp_com_id_i = com_id_i;
            temp_com_id_i(j) = [];
            
            k = 1;
            while k <= length(temp_com_id_i)
                if adj( cur_node_id, temp_com_id_i(k) ) == 1
                    temp_com_id_i(k) = [];
                else
                    k = k + 1;
                end
            end
            
            for k = 1:remain_num
                
                rand_index = floor(rand*length(temp_com_id_i)) + 1;
                
                conn_node_id = temp_com_id_i( rand_index );
                
                if adj( cur_node_id, conn_node_id ) == 1
                    fprintf('Error: %d and %d have alreadly connected!!!\n',cur_node_id,conn_node_id);
                    exit;
                end
                
                adj( cur_node_id, conn_node_id ) = 1;
                adj( conn_node_id, cur_node_id ) = 1;
                
                z_in_arr(cur_node_id) = z_in_arr(cur_node_id) + 1;
                z_in_arr(conn_node_id) = z_in_arr(conn_node_id) + 1;
                
                temp_com_id_i( rand_index ) = [];
                
            end
        end
    end
end

for i = 1:numCluster
    
    com_id_i = community{i};
    
    ext_node_id = [];
    for j = 1:numCluster
        if j~=i
            ext_node_id = [ ext_node_id, community{j} ];
        end
    end
    
    for j = 1:numNode
        
        cur_node_id = com_id_i(j);
        
        temp_ext_node_id = ext_node_id;
        
        k = 1;
        while k <= length(temp_ext_node_id)
            
            if adj( cur_node_id, temp_ext_node_id(k) ) == 1
                temp_ext_node_id(k) = [];
            else
                k = k + 1;
            end
        end
        
        exist_inter_link = (numCluster-1)*numNode - length(temp_ext_node_id);
        
        if z_out_arr(cur_node_id) ~= exist_inter_link
            fprintf('Error has occured: z_out_arr(cur_node_id) ~= exist_inter_link!!!\n');
            exit;
        end
        
        zout_left = zout - z_out_arr(cur_node_id);
        
        flag = 0;
        
        for k = 1:zout_left
            
            v = 1;
            while v <= length(temp_ext_node_id)
                if z_in_arr(temp_ext_node_id(v)) == zout
                    temp_ext_node_id(v) = [];
                else
                    v = v + 1;
                end
            end
            
            if isempty(temp_ext_node_id)
                flag = 1;
                break
            end
            
            rand_index = floor(rand*length(temp_ext_node_id)) + 1;
            
            conn_ext_node_id = temp_ext_node_id( rand_index );
            
            if adj( cur_node_id, conn_ext_node_id ) == 1
                fprintf('Error: %d and %d have alreadly connected!!!\n',cur_node_id,conn_ext_node_id);
                exit;
            end
            
            adj( cur_node_id, conn_ext_node_id ) = 1;
            adj( conn_ext_node_id, cur_node_id ) = 1;
            
            z_out_arr(cur_node_id) = z_out_arr(cur_node_id) + 1;
            z_out_arr(conn_ext_node_id) = z_out_arr(conn_ext_node_id) + 1;
            
            temp_ext_node_id( rand_index ) = [];
            
        end
        
        if flag==1
            
            remain_num = zout - z_out_arr(cur_node_id);
            
            temp_ext_node_id = ext_node_id;
            temp_ext_node_id(j) = [];
            
            k = 1;
            while k <= length(temp_ext_node_id)
                if adj( cur_node_id, temp_ext_node_id(k) ) == 1
                    temp_ext_node_id(k) = [];
                else
                    k = k + 1;
                end
            end
        
            rand_index = floor(rand*length(temp_ext_node_id)) + 1;
            
            conn_ext_node_id = temp_ext_node_id( rand_index );
            
            if adj( cur_node_id, conn_ext_node_id ) == 1
                fprintf('Error: %d and %d have alreadly connected!!!\n',cur_node_id,conn_ext_node_id);
                exit;
            end
            
            adj( cur_node_id, conn_ext_node_id ) = 1;
            adj( conn_ext_node_id, cur_node_id ) = 1;
            
            z_out_arr(cur_node_id) = z_out_arr(cur_node_id) + 1;
            z_out_arr(conn_ext_node_id) = z_out_arr(conn_ext_node_id) + 1;
            
            temp_ext_node_id( rand_index ) = [];

        end

    end
end
