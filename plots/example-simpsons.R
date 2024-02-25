library(MASS)
library(tidyverse)
library(ggthemes)

# Code adopted from https://www.r-bloggers.com/2020/11/simpsons-paradox-and-misleading-statistical-inference/
# Daniel Klotz, 2023

set.seed(3)
### build the g1
mu<-c(6.5,2.5)
sigma<-rbind(c(1,0.4),c(0.4,1) )
g1<-as.data.frame(mvrnorm(n=1000, mu=mu, Sigma=sigma))
g1$group<-c("machine learning")
### build the g2
mu<-c(4.5,4.5)
g2<-as.data.frame(mvrnorm(n=1000, mu=mu, Sigma=sigma))
g2$group<-c("statistics")

### build the g3
mu<-c(2.5,6.)
g3<-as.data.frame(mvrnorm(n=1000, mu=mu, Sigma=sigma))
g3$group<-c("surface hydrology")

# the combined data of all three groups
df <- rbind(g1,g2,g3)

# make plots 
f_size = 9
plt1 <- ggplot(data = df, aes(x=V1, y=V2)) + 
  geom_point(size=0.5, color="black") + 
  #geom_smooth(method='lm', color="red", se=F) +
  labs(title="Impact of studying on grades",
       subtitle="overall relationship",
       y="grade",
       x="preparation hours") +
  theme_tufte() + 
  theme(panel.grid = element_line(colour="#cccccc", size=0.3),
        panel.border = element_rect(fill=NA, colour="#cccccc"),
        text = element_text(size=f_size))
plt2 <- ggplot(data = df, aes(x=V1, y=V2, group=group, col=group)) + 
  geom_point(size=0.5) + 
  #geom_smooth(method='lm', col='black', se=F) +
  scale_color_viridis_d("class:") +
  labs(subtitle="class relationship",
       y="grade",
       x="preparation hours") +
  theme_tufte() + 
  theme(panel.grid = element_line(colour="#cccccc", size=0.2),
        panel.border = element_rect(fill=NA, colour="#cccccc"),
        text = element_text(size=f_size),
        legend.position="bottom",
        legend.spacing.x = unit(0.005, 'cm'))
#
library(patchwork)
plt1 / plt2 #+ plot_layout(widths=c(9,10))

ggsave(filename = "./Desktop/example-simpsons.pdf", height= 10, width = 8.3, units="cm", dpi=500)

