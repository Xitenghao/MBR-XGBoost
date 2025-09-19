library(MASS)
library(ggplot2)
library(ggisoband)
library(ggpointdensity)
library(viridis)
library(RColorBrewer)
library(sf)
library(latex2exp)
library(gridExtra)
library(ggpubr)

kde2dc = function (x, y, h, n = 50, lims = c(range(x), range(y))) 
{
    nx <- length(x)
    n <- rep(n, length.out = 2L)
    gx <- seq.int(lims[1L], 1.4*lims[2L], length.out = n[1L])
    gy <- seq.int(lims[3L], 1.4*lims[4L], length.out = n[2L])
    h <- c(bandwidth.nrd(x), bandwidth.nrd(y))
    #h <- h/4
    ax <- outer(gx, x, "-")/h[1L]
    ay <- outer(gy, y, "-")/h[2L]
    z <- tcrossprod(matrix(dgamma(ax,1), , nx), matrix(dgamma(ay,1), 
        , nx))/(nx * h[1L] * h[2L])
    list(x = gx, y = gy, z = z)
}


data <- read.table("data_x_ac.csv", encoding='gbk', header = TRUE, sep = ",", quote = "\"", dec = ".")
density_plot = function (x1)
{
    x = data[[x1]]
    y = data$se_mean
    f <- kde2dc(x, y, n = 1000, h = c(width.SJ(x), width.SJ(y)))
    
    x <- f$x
    y <- f$y
    z <- f$z
    
    density_data <- expand.grid(x = x, y = y)
    density_data$z <- as.vector(z)
    if (x1 == 'A.T'){
        label_x <- 'A/T'   
    }
    else if(x1=='FDSi') {
        label_x <- TeX('$FD_{Si}$')
    }
    else if(x1=='crystal.size.μm.') {
        label_x <- 'CS'
    }
    else if(x1=='reaction.temp..C.') {
        label_x <- 'RT'
    }
    else if(x1=='WHSV.h.1.') {
        label_x <- 'WHSV'
    }
    else{label_x <- x1}
    my_colormap <- colorRampPalette(rev(brewer.pal(11,'Spectral')))(32)
    # my_colormap <- colorRampPalette(c('#FBFCFC','#07366E'))(32)
    ggplot() + 
        geom_tile(data = density_data, aes(x=x, y=y, fill=z)) +
        scale_fill_gradientn(colours = my_colormap, name = "Density") + 
        labs(x = label_x, y = "Mean value of Sethylene") + 
        theme_minimal() +
        theme(legend.position = "none",
              axis.text.x = element_text(size = 11),
              axis.text.y = element_text(size = 11),
              legend.key.height = unit(4.3, 'cm'),
              legend.key.width = unit(0.6, 'cm'),
              legend.title = element_text(size = 10),
              legend.text = element_text(size = 10),
              panel.grid = element_blank(),
              panel.background = element_blank()
              )
}

columns = c('MDa', 'MDb', 'MDc', 'Mdi', 'A.T', 'FDSi',  'crystal.size.μm.', 'reaction.temp..C.', 'WHSV.h.1.')
plot_list <- list()
for (i in 1:9) {
    new_x_labels <- c('MDa', 'MDb', 'MDc', 'Mdi','A/T', '$\text{FD}_{\text{Si}}$', 'CS', 'RT', 'WHSV')
    plot_list[[i]] <- density_plot(columns[i])
}

grid <- ggarrange(plot_list[[1]], plot_list[[2]], plot_list[[3]],
                  plot_list[[4]], plot_list[[5]], plot_list[[6]],
                  plot_list[[7]], plot_list[[8]], plot_list[[9]],
                  ncol = 3, nrow = 3,
                  common.legend = T,
                  legend = "right") 
ggsave("...", grid, width = 9, height = 9, dpi = 300)


