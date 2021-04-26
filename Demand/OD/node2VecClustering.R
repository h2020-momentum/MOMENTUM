library(lubridate)
library(tidyverse)
library(pals)
library(ape)


############################################################################################################
##SCRIPT DETAILS##
############################################################################################################
#
# The objective of this script is to group a set of OD Matrices according to a specific similarity measure
# using a new approach based on Graph Embeddings and concretegly on Node2Vec.
#
#Input: File with similarity measures in format
#Output: File with clusters for OD Matrix according to the approach based on Graph Embeddings. The cut-off threshold
#        is set by the variable cluster_height_cut


Sys.setlocale("LC_TIME", "English")


#####################################################
# CREATION OF THE OD MATRIX SIMILARITY GRAPH
#####################################################

#Read file
data = read.csv("SimilarityMeasure.csv")


#Create columns with complete data to facilitate visualization
data[data$day1 < 10,"date1" ] = paste0(data$month1[data$day1 < 10],"0",data$day1[data$day1 < 10]) 

data[data$day1 >= 10,"date1" ] = paste0(data$month1[data$day1 >= 10],data$day1[data$day1 >= 10])


data[data$day2 < 10,"date2" ] = paste0(data$month2[data$day2 < 10],"0",data$day2[data$day2 < 10]) 

data[data$day2 >= 10,"date2" ] = paste0(data$month2[data$day2 >= 10],data$day2[data$day2 >= 10])

# Since we are creating an undirected graph, redundant day-to-day comparison are filtered
dataNode2Vec = subset(data, (month1 <= month2) & (day1 < day2), select = c("date1","date2","similarity"))

# Similarity is normalized between 0 and 1
dataNode2Vec2 = mutate(dataNode2Vec, normSim = (max(similarity) - similarity)/(max(similarity)- min(similarity)))

# Selection of those nodes whose similarity is higher than the percentile 70
dataNode2Vec = subset(dataNode2Vec2, normSim > quantile(dataNode2Vec2$normSim,0.7)) 

# Creation of the edg file with the edges of the graph
write.table(dataNode2Vec[,c(1,2,4)],"similarity_edges.edg", sep="\t",quote = FALSE, dec = ".", row.names = FALSE, col.names = FALSE)


##################################################
# Generation of node embeddings using pecanpy tool 
##################################################

system("pecanpy --input similarity_edges.edg --output similarity_nodes.emb --mode SparseOTF --weighted --verbose --p 1 --q 20 --num-walks 500 --walk-length 10",intern = TRUE)

##################################################
# Generation of clusters
##################################################

dataEmb = read.table("similarity_nodes.emb", header = FALSE, skip = 1, sep = " ", dec = ".")

#Variable to indicate the cut-off threshold for the hierarchical clustering
cluster_height_cut = 8.5
 
h_clust = hclust(dist(dataEmb[, 2:129]), method="ward.D")

h_clust$labels = format(ymd(dataEmb$V1),"%a%y-%m-%d") 

clus = cutree(h_clust, h = cluster_height_cut)



##################################################
# Visualization of clusters
##################################################

colors = palette.colors(length(unique(clus)), palette = "alphabet")
plot(as.phylo(h_clust), type = "fan", tip.color = colors[clus],
      label.offset = 1, cex = 0.7)
 

leg_labels = sapply(c(1:length(unique(clus))),function(x){paste0("Clust ", x)})
 

legend(x=60, y=0, yjust= 0.5, legend=leg_labels, fill = colors,  cex=0.8, text.width = 2, bty="n" , y.intersp = -0.5, x.intersp = 0.2 )


##################################################
# Output file writting
##################################################

output = data.frame(date = dataEmb$V1, cluster = clus)

write.csv(output,"clusters_info.csv", row.names = FALSE, quote = FALSE)
 