#Packages 

using Statistics
using Flux
using DataFrames
using MAT

#Define the file paths

folder = "C:\\Users\\Dell\\Downloads\\Neural\\Relevant"
foldertrain = joinpath(folder, "Train")
foldertest = joinpath(folder, "Test")
train_folder = joinpath(foldertrain,"data.mat")

#Creating DataFrames

filetest_1 = DataFrame(V = [], I = [], Temperature = [], Capacity = [], Energy = [], SOC = [])
filetest_2 = DataFrame(V = [], I = [], Temperature = [], Capacity = [], Energy = [], SOC = [])
filetest_3 = DataFrame(V = [], I = [], Temperature = [], Capacity = [], Energy = [], SOC = [])
filetest_4 = DataFrame(V = [], I = [], Temperature = [], Capacity = [], Energy = [], SOC = [])
data_train = DataFrame(V = [], I = [], Temperature = [], Capacity = [], Energy = [], SOC = [])
Error = DataFrame(File = [], Max = [],Mean = [],RMSE = [])


#Opening and reading MAT files for testing

function openmatfiles_test!()
    for i in 1:4
        file = matopen(foldertest*"\\data"*string(i)*".mat")
        y_test = read(file, "Y")
        x_test = read(file, "X")
        y_test = DataFrame(y_test,:auto)
        x_test = DataFrame(x_test,:auto)
        y_test = dropmissing(y_test)
        x_test = dropmissing(x_test)
        if i == 1
            global filetest_1 = vcat(x_test,y_test)
        elseif i == 2
            global filetest_2 = vcat(x_test,y_test)
        elseif i == 3
            global filetest_3 = vcat(x_test,y_test)
        else
            global filetest_4 = vcat(x_test,y_test)
        end
    end
end

#Opening and reading MAT files for training
function openmatfiles_train!()
    matfile = matopen(train_folder)
    global y_train = read(matfile, "Y")
    global x_train = read(matfile, "X")
    close(matfile)

    df_y_train = DataFrame(y_train,:auto)
    df_x_train = DataFrame(x_train,:auto)
    df_y_train = dropmissing(df_y_train)
    df_x_train = dropmissing(df_x_train)

    data_train = vcat(df_x_train,df_y_train)

    x_train = Float32.(Matrix(data_train[1:5, :]))
    y_train = data_train[6,:]
    y_train = Vector(y_train)
    y_train = y_train[:,:]
    y_train =  transpose(y_train)

end


#Defining Architecture and Parameters, Loading data and training the model

function define_parameters!(num_hidden_units,first_layer,last_layer,batch,epochs,repetitions)

    global model = Chain(
                    Dense(first_layer, num_hidden_units, tanh),
                    Dense(num_hidden_units, num_hidden_units, leakyrelu),
                    Dense(num_hidden_units, last_layer)
                )

    loss(x, y) = Flux.mse(model(x), y)

    local data_input = x_train
    local data_output = y_train

    local dataset = Flux.Data.DataLoader((data_input,data_output), batchsize = batch, shuffle=true)
    local parameters = Flux.params(model)
    local opt = Flux.Optimise.ADAM(0.01,(0.9, 0.95), eps(typeof(0.01)))

    for f in 1:repetitions
        for i in 1:epochs

            Flux.train!(loss, parameters, dataset, opt)
            println("Epoch "*string(i))
        end
    end
end

#Testing 

function testing!(testing_file)
    local Test_data = Float32.(testing_file)
    local predictions_1 = []
    local error_test = []
    local error_testsqr = []
    local Error = DataFrame()


    for i in 1:size(Test_data,2)
        local prediction = model(Test_data[1:5,i])
        prediction =  prediction[1]
        local scaled = prediction
        push!(predictions_1,scaled)
    end


    # Calculate the errors
    for i in 1:size(Test_data,2)
        local temp = predictions_1[i]
        local error = temp[1] .-  Test_data[6,i]
        push!(error_test,error)
        errorsqr = error^2
        push!(error_testsqr,errorsqr)
    end

    # Compute RMSE, MAE, and MAX errors
    rmse_test = sqrt(mean(error_testsqr))
    mae_test = mean(abs.(error_test))
    max_test = maximum(abs.(error_test))

    Error = DataFrame(Max = max_test,Mean = mae_test,RMSE = rmse_test)

    println(Error)

end


function main()

    local x_train = []
    local y_train = []

    #Call function to open test and train files
    openmatfiles_test!()
    openmatfiles_train!()

    #Call function to define parameters and hyperparameters
    define_parameters!(55,5,1,32,150,3)


    #Call file for testing
    testing!(filetest_1)
    testing!(filetest_2)
    testing!(filetest_3)
    testing!(filetest_4)

end


main()

