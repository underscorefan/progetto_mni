plot_2d_kmeans<-function(data, x, y, to_fact, backg = "grey69", plot_col = "white", main = ""){
  par(bg = backg, col = plot_col, col.lab = plot_col, col.axis = plot_col)
  f <- names(table(data[[to_fact]]))
  n_colors <- length(f) 
  a_colors <- palette(rainbow(n_colors))
  plot(data[[x]],data[[y]], col = backg, xlab = "x", ylab = "y", main = main, col.main = plot_col)
  axis(side = 1, col.ticks = plot_col, col = plot_col)
  axis(side = 2, col.ticks = plot_col, col = plot_col)
  for (i in 1:n_colors) {
    pnts<-data[data[[to_fact]] == f[i],]
    points(pnts[[x]],pnts[[y]], col = a_colors[i])
  }
  return(a_colors)
}

add_centroids<-function(data, x, y, colo){
  ps<-length(data[[x]])
  for(i in 1:ps){
    pnts<-data[i,]
    points(pnts[[x]],pnts[[y]], col = colo[i], pch = 8, cex = 2)
  }
}
# points ../mpi_code/final_table.csv
# centroids ../mpi_code/centroids.csv
plot_all<-function(points, centrs, backg = "grey69", plot_col = "white", main = ""){
  add_centroids(fetch_path(centrs), "V1", "V2", plot_2d_kmeans(fetch_path(points), "V1", "V2", "V3", backg = backg, plot_col = plot_col, main = main))
}

fetch_path<-function(p){
  read.csv2(p, header = F, stringsAsFactors = F, sep = ",")
}