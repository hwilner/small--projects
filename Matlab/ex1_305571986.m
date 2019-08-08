function ex1()
    % Clear the command prompt, close all windows and clear the workspace
    clc; close all; clear;
    
    w = [0.6,-0.2]; % The connections weights. Initialized by random weights.
    bias = 0.9; % The output neuron threshold
    learningRate = 0.2; % The learning rate
    
    % Maximum iterations for the main loop. After that the algorithm ends
    % without finding a solution
    MAX_TURNS = 1000;

    % 4 different input files. Run the results for each one of them. 
    dataFiles = {'binaryClassData1.txt', 'binaryClassData2.txt', 'binaryClassData3.txt','binaryClassData4.txt'};
    
    % You choose which input file to load manualy: dataFiles{x} for 
    % file number x (x={1,2,3,4}
    x=load(dataFiles{4});
    
    % Seperate the input into x1,x2 and d, the desire output
    x1 = x(:,1);
    x2 = x(:,2);
    t = x(:,3);
    % Initialize the errors vector as MAX_TURNS zeros
    errors = zeros(1,MAX_TURNS);
    
    % Initialize the turn number as 0. Each full iteration over all the
    % samples is one turn.
    turn = 0;
    % errors_exist=true means we should run over the samples again. It
    % means that at least one sample's output isn't the desire output
    errors_exist = true;
    % Run over the samples while there is at least one error, or until
    % MAX_TURNS turns
    while (errors_exist && turn<MAX_TURNS)
        turn=turn+1;
        errors_exist = false;
        errors_num = 0;
        for sample_ind=1:length(x1)
            input_vec = [x1(sample_ind),x2(sample_ind)];
           
            % 1) Change the next code line for calculating the 
            % perceptrons's net input for sample i.
            net_input = input_vec * w' + bias;
            
            
            % 2) Change the next code line for calculating the output for
            % sample Use the equation y=z(net_input)
            % Where z gives 1 if net_input >0 , and otherwise zero.
            output = net_input>0;
            
            % Check if the output is equal to the desire output
            if (output~=t(sample_ind))
               
                % 3) Change the next 3 code lines to calculate the new
                % values for w(1) and w(2) and bias, according to the perceptron's
                % learning rule. Use the equation.
                w=w+learningRate.*(t(sample_ind)-output).*input_vec;
                bias = bias+learningRate*(t(sample_ind)-output);
                
                % Mark that at least one error exist, so after the algorithm will
                % continue to go over the samples, it will go over all of
                % them agian
                errors_exist = true;
                % Count the errors number for this turn
                errors_num = errors_num + 1;
            end
        end
        
        % Save the errors number in this turn into errors(turn)
        errors(turn) = errors_num;
    
    end
    
    % Plot the graph - the input dots with the seperating line
    plot_graph(x1,x2,t,w,bias);
    % Plot the errors number
    figure();
    bar(1:turn,errors(1:turn));
    title('Number of errors by turn');

    % Inner function for plotting the graph
    function plot_graph(x1,x2,d,w,bias)
        % For plotting the seperating line, we need to calculte the first
        % line's dot, where we want to start plotting it,
        % [linex(1),liney(1)] and until where we want to plot it, [linex(2), liney(2)]
        % calc_y is a function that calculates x2 from x1.
        linex = [min(x1),max(x1)]; 
        liney = [calc_y(linex(1),w,bias) calc_y(linex(2),w,bias)];

        % Open a figure
        figure();
        % Plot the plots on the same figure
        hold on
        % Plot the input dots 
        % Make sure you undestand the use of logical indices here.
        scatter(x1(d==1),x2(d==1),'ro'); % plot the input dots where d=1
        scatter(x1(d==0),x2(d==0),'g+'); % plot the input dots where d=0
        % Plot the seperating line
        line(linex,liney);
        legend('1','0');
        ylim([min(x2),max(x2)]); 
        title(['The separeting line the perceptron has found, w=[', num2str(w(1)), ',', num2str(w(2)), ',], b=',num2str(bias)]);
    end

    function y=calc_y(x,w,bias)
        % 4) This funtcion should calculate the Y value for the input
        % x value, assuming x and y are on the line separating the 
        % two classes. Use the explicit line equation we showed in class.
        y = (((-w(1)).*x)-bias)./w(2);
    end
    
end

