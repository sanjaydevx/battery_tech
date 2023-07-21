using DataFrames
using CSV
using Plots
using DataStructures
using BenchmarkTools
using Statistics

#Reading data and loading onto a csv file
data = CSV.read("D:\\workspace\\github\\battery_tech\\data\\batterydata.csv", DataFrame, header = true,);
data = dropmissing(data)

#Global Variables
clumps_current = []

#Global Dataframes
dataframe = DataFrame(current=[],soc_change=[],full_charge=[])

#singling out SOC column
soc = data[:,7]
current = data[:,6]
l = length(soc)
x_axis = collect(1:l)

#Finding instances of current flow
function instances(input,output,soc)
    local i = 1
    local sum = 0
    local high = 0
    local low = 0 

    for i in 1:length(input)
        if i < length(input)
            if input[i]!=0
                if sum ==0
                    high = soc[i]
                end
                sum = sum + input[i]
                if input[i+1]==0
                    low = soc[i]
                    if high != low
                        push!(dataframe.current,sum)
                        push!(dataframe.soc_change,abs(high-low))
                        push!(dataframe.full_charge,100*sum/abs(high-low))
                    end
                sum = 0 
                end
            end
        end
    end
end

#Main

function main()
    instances(current,clumps_current,soc)

    println(dataframe)
end

main()

#plotting SOC over time
p1 = gui(plot(collect(1:length(dataframe.current)),dataframe.full_charge,label = "full charge"))

png("D:\\workspace\\workingdirectory\\p1")
