
using Statistics
using Flux
using DataFrames
using FileIO
using MAT



# Define the file paths
folder = "C:\\Users\\Dell\\Downloads\\Neural\\Relevant"
foldertrain = joinpath(folder, "Train")
foldertest = joinpath(folder, "Test")
train_folder = joinpath(foldertrain,"data.mat")


#Creating Dataframes

filetest_1 = DataFrame(V = [], I = [], Capacity = [], Temperature = [], Something = [], SOC = [])
filetest_2 = DataFrame(V = [], I = [], Capacity = [], Temperature = [], Something = [], SOC = [])
filetest_3 = DataFrame(V = [], I = [], Capacity = [], Temperature = [], Something = [], SOC = [])
filetest_4 = DataFrame(V = [], I = [], Capacity = [], Temperature = [], Something = [], SOC = [])
data_train = DataFrame(V = [], I = [], Capacity = [], Temperature = [], Something = [], SOC = [])


#Opening and reading MAT files for testing
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
        global filetest_3 = vcat(x_test,y_test)
    end
end

# Opening and reading MAT files for training
matfile = matopen(train_folder)
y_train = read(matfile, "Y")
x_train = read(matfile, "X")
close(matfile)


df_y_train = DataFrame(y_train,:auto)
df_x_train = DataFrame(x_train,:auto)
df_y_train = dropmissing(df_y_train)
df_x_train = dropmissing(df_x_train)

#Concatenate 

data_train = vcat(df_x_train,df_y_train)

#Defining NN architecture 

num_hidden_units = 55

model = Chain(
                Dense(3, num_hidden_units, tanh),
                Dense(num_hidden_units, num_hidden_units, leakyrelu),
                Dense(num_hidden_units, 1)
            )

#Defining Epochs, batch size and learning rate
epochs = 50
batch_size = 32
learning_rate = 0.01

#Defining optimizer and loss function

opt = Flux.Optimise.ADAM(learning_rate)
loss(x, y) = Flux.mse(model(x), y)

#Converting data into arrays 
x_train_array = Float32.(Matrix(data_train[1:3, :]))
y_train_array = data_train[6,:]
y_train_array = Vector(y_train_array)
y_train_array = y_train_array[:,:]
y_train_array =  transpose(y_train_array)

#data = [(x_train_array,y_train_array)]

dataset = Flux.Data.DataLoader((x_train_array,y_train_array), batchsize=batch_size, shuffle=true)

#Training the model

parameters = Flux.params(model)

for i in 1:epochs
    Flux.train!(loss, parameters, dataset, opt)
end

#Testing

testfile_4 = Float32.(data_train)

predictions_1 = []

for i in 1:size(testfile_4,2)
    prediction = model(testfile_4[1:3,i])
    push!(predictions_1,prediction)
end 

CSV_df = DataFrame(SOC_predicted = predictions_1)

using CSV
CSV.write(joinpath(folder,"SOC_predicted.csv"), CSV_df)

#=
Target_1 = testfile_4[6, :]


error_test = []
error_testsqr = []
# Calculate the errors
for i in 1:size(testfile_4,2)
    temp = predictions_1[i]
    error = temp[1] .-  Target_1[i]
    push!(error_test,error)
    errorsqr = error^2
    push!(error_testsqr,errorsqr)

end

# Compute RMSE, MAE, and MAX errors
rmse_test = sqrt(mean(error_testsqr))
mae_test = mean(abs.(error_test))
max_test = maximum(abs.(error_test))

println(rmse_test)
println(mae_test)
println(max_test)
=#
