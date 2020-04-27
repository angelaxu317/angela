######
#R -Statistics
######

#Descriptive Statistics
# one method of obtaining descriptive statistics is to use the sapply() function
#with a specified summary statistic

#get means for variables in data frame mydata
#excluding missing values
sapply(mydata, mean, na.rm=TRUE)
#possible functions used in sapply include mean, sd,var,min,max, 
#median,range and quantile

#R functions designed to provide a range of descriptive statistics at once
#mean, median, 25th and 75th quartiles, min,max
summary(mydata)

#tukey min, lower-hinge, median, upper-hinge, max
fivenum(x)

#using the Hmisc package
library(Hmisc)
describe(mydata)
#n,nmiss,unique,mean,5,10,25,50,75,90,95th percentiles
#5 lowest and 5 highest  scores

#using pastecs package
library(pastecs)
stat.desc(mydata)
#nbr.val, nbr,null, nbr.na, min,max, range,sum
#median, mean, SE.mean, CI.mean, var std.dev, coef.var

#using the psych package
library(psych)
describe(mydata)
#item name, item number, nvalid, mean, sd,
#median, mad, min,max,skew, kurtosis, se

#summary statistics by grouping variable
library(psych)
describe.by(mydata, group, ...)

#doBy package provides much of the functionality of SAS PROC SUMMARY
library(doBy)
summaryBy(mpg +wt ~ cyl +vs, data=mtcars,
          FUN=function(x){c(m=mean(x), s=sd(x))})
#produces mpg.m wt.m mpg.s wt.s for each
#combination of the levels of cyl and vs


###Frequencies and Crosstabs
#create frequency and contingency tables from categorical variables
#tests if undependence, measures of association 
#methods for graphically displaying results

#generating frequency tables A,B, C represent categorical variables
#generate frequency tables using table() function
#tables of proportions using prop.table()
#marginal frequencies using margin.table()

#2-way Frequency Table
attach(mydata)
mytable<-table(A,B) #A will be rows, B will be columns
mytable #print table

margin.table(mytable,1) #A frequencies (summed over B)
margin.table(mytable,2) #B frequencies (summed over A)

prop.table(mytable) #cell percentages
prop.table(mytable,1)#row percentages
prop.table(mytable,2)#column percentages

#table() can also generat multidimensional tables based on 
#3 or more categorical variables
#use ftable() function to print the results more attactively

# 3-way frequency table
mytable<-table(A,B,C)
ftable(mytable)
#table ignores missing values
# to include NA as a category in counts, include the table option exclude =NULL
#if the variable is a vector
#if the variable is a factor you have to create a new factor using 
#newfactor<-factor(oldfactor, exclude=NULL)

#xtabs() function allows you to creat crosstabulations using formula style 
#input

#3-way Frequency table
mytable<-xtabs(~A+B+C, data=mydata)
ftable(mytable) #print table
summary(mytable) #chi-square test of independence
#CrossTable() function in the gmodels package produces crosstabulations modeled
#after Proc FREQ in SAS or CROSSTABS in SPSS

#2-way cross tabulation
library(gmodels)
CrossTable(mydata$myrowvar, mydata$mycolvar)
#there are options to report percentages(row,column,cell), 
#specifydecimal places, produce Chi-square, Fisher and McNemar tests of 
#independence
#report expected and residual values (pearson, standardized, adjusted 
#standardized)
#include missing values as valid, annotate with row and column titles

#tests of independence
#Chi-Square Test
# for 2-way tables can use chisq.test(mytable) to test independence of the
#row and column variable
#by default, the p-value is calculated from the asymptotic chi-squared 
#distribution of the test statistic (Can be derived via Monte Carlo simulation)

#fisher.test(x) provides an exact test of independence
#x is a two dimensional contingency table in matrix form

#mantelhaen.test(x)

#use loglm() function in MASS package to produce log-linear models
#assume we have a 3-way contingency table baased on variables A,B, and C
library (MASS)
mytable<-xtabs(~A+B+C, data=mydata)
#mutual independence tests: A,B,and C are pairwise independent
loglm(~A+B+C, mytable)
#partial independence tests:A is partially independent of B and C 
#i.e., A is independent of the composite variable BC
loglin(~A+B+C+B*C, mytable)
#conditional independence:A is independent of B, given C
loglm(~A+B+C+A*C+B*C, mytable)
#no three -way interaction
loglm(~A+B+C+A*B+A*C+B*C, mytable)


### Correlations
#use cor() to produce correlations
#cov() to produce covariances
#cor(X, use=, method=)

#correlations/covariances among numberic variales in data fram mtcars
#use listwise deletion of missing data
cor(mtcars, use="complete.obs", method="kendall")
cov(mtcars, use="complete.obs")

#correlations with significance levels
library(Hmisc)
rcorr(x, type="pearson") #type can be pearson or spearman

#mtcars is a data frame
rcorr(as.matrix(mtcars))

#cor(X,Y) or rcorr(X, Y) generate correlations between columns X and 
#the columns of Y
#correlation matrix from mtcars
x<-mtcars[1:3]
y<-mtcars[4:6]
cor(x,y)

#othe types of correlations
#polychoric correlation
#x is a contingency table of counts
library(polycor)
polychor(x)

#heterogeneous correlations in one matrix
#pearson(numeric-numeric)
#polyserial(numeric -ordinal)
#polychoric(ordinal-ordinal)
#x is a data frame with ordered factors and numeric variables
library (polycor)
hetcor(x)

#partial correlations
library(ggm)
data(mydata)
pcor(c("a","b","x","y","z"),var(mydata))
#partial corr between a and b controlling for x,y,z

###T-Tests
#t.test() the default assumes unequal varianc 
#and applies the Welsh df modification
#var.equal=TRUE option to specify equal variances
#and a pooled variance estimate
#to specify a one tailed test alternative="less" or alternative= "greater"


#independent 2-group t-test
t.test(y~x) #y is numeric and x is a binary factor

t.test(y1,y2) #where y1 and y2 are numeric

#paired t-test
t.test(y1,y2,paired=TRUE) #y1,y2 are numeric

#one sample t-test
t.test(y,mu=3) #H0: mu=3


###Nonparametric Test of Group differences
#provides functions for carrying out Mann-Whitney U
#Wilcoxon Signed Rank, Krushal Wallis, Friedan tests

#indenpendent 2-group Mann-Whitney U Test
wilcox.test(y~A) #y is numeric and A is a binary factor

wilcox.test(y,x) #y and x are numeric

#indenpendent 2-group Wilcoxon Signed Rank
wilcox.test((y1, y2, paired=TRUE)) #y1 and y2 are numeric

#kruskal Wallis Test one way anova by ranks
kruskal.test(y~A) #y1 is numeric and A is a factor

#Randomized Block design -Friedman Test
friedman.test(y~A|B)
#y are the data values A is a grouping factor B is a blocking factor

#to specify a one tailed test alternative="less" or alternative= "greater"


###Multiple (linear) Regression

#Fitting the model
#Multiple linear regression example
fit<-lm(y~x1+x2+x3,data= mydata)
summary(fit) #show results

#Other useful functions
coefficients(fit) #model coefficients
confint(fit, level=0.95) #CI for model parameters
fitted(fit) #predicted values
residuals(fit) #residules
anova(fit) #anova table
vcov(fit) #covariance matrix for model paameters
influence(fit) #regression diagnostics

#Diagnostic Plots
#provide checks for heteroscedasticity, normality and influential observations
layout(matrix(c(1,2,3,4),2,2)) #optional 4 graphs/page
plot(fit)

#comparing models
#compare nested models witht the anova()function
fit1<-lm(y~x1+x2+x3+x4, data=mydata)
fit2<-lm(y~x1+x2)
anova(fit1,fit2)

#cross validation
#do a k-fold cross-validation using cv.lm() function in DAAG package
library(DAAG)
cv.lm(df=mydata,fit,m=3) #3 fold cross-validation

#sum the MSE for each fold, divided by the number of observations 
#and take the square root to get the cross-validated standard error of estimate

#assess R2 shrinkage via K-fold cross-validation
#using crossval() function from bootstrap package

#assessing R2 shrinkage using 10-fold cross-validation
fit<-lm(y~x2+x2+x3,data=mydata)

library(bootstrap)
#define fuctions
theta.fit<-function(x,y){lsfit(x,y)}
theta.predict<-function(fit,x){cbind(1,x) %*%fit$coef}

#matrix of predictors
x<-as.matrix(mydata[c("x1","x2","x3")])
#vector of predicted values
y<- as.matrix(mydata[c("y")])

results<-crossval(X,y,theta.fit, theta.predict,ngroup=10)
cor(y,fit$fitted.values)**2  #raw R2
cor(y,results$cv.fit)**2 # cross-validated R2

# Variable Selection
#selecting asubset of predictor variables fro a larger set
#use stepwise select (forward, backward, both) using stepAIC() function 
#from MASS package

library(MASS)
fit<-lm(y~x1+x2+x3,data=mydata)
step<- stepAIC(fit,direction = "both")
step$anova #display results

#perform all-subsets regression using the leaps() function from leaps package
#nbest indicates the number of subsets of each size to report
#10 best models will be reported 
#for each subset size (1 predictor, 2 predictors,etc.)

#all subsets regression
library(leaps)
attach(mydata)
leaps<-regsubsets (y~x1+x2+x3+x4, data=mydata, nbest=10)
summary(leaps) #view resutls
#plot a table of models showing variables in each model
#models are ordered by the selection statistic
plot(leaps,scale="r2")
#plot statistic by subset size
library(car)
subsets(leaps,statistics="rsq")


### Relative importance
# relaimpo package provides measures of relative importance for each of the 
#predictors in the model calc.relimp details on the four measures of 
#relative importance provided

library(relaimpo)
calc.relimp(fit,type=c("lmg","last","first","pratt"), rela=TRUE)

#bootstrap Measures of relative importance (1000 samples)
boot<-boot.relimp(fit, b=1000,type=c("lmg","last","first","pratt"),
                  rank=TRUE, diff=TRUE,rela=TRUE)
booteval.relimp(boot) #print result
plot(booteval.relimp(boot, sort=TRUE)) #plot result

### Regression Diagnostics
#assime we are fitting a multiple linear regression on MTCARS data
library(car)
fit<-lm(mpg~disp+hp+wt+drat,data-mtcars)

#Outliers
outlierTest(fit) #bonferonni p-value for most extreme obs
qqplot(fit, main="QQ Plot") #qq plot for studentized residuale
leveragePlots(fit) #leverage plots

#influential observations
#add variable plots
av.Plots(fit)
#cook's D plot
#identify D values >4/(n-k-1)
cutoff<- 4/((nrow(mtcars)-length(fit$coefficients)-2))
plot(fit,which=4, cook.levels=cutoff)
#influence plot
influencePlot(fit,id.method="identify",main="influence Plot",
              sub="Circle size is proportial to Cook's Distance")

#NON_normatlity
#normality of residuals
#qq plot for studentized resid
qqPlot(fit, main="QQ Plot")
#distribution of studentized residuals
library(MASS)
sresid<-studres(fit)
hist(sresid, freq = FALSE,
     main= "Distribution of Studentized Residuals")
xfit<-seq(min(sresid),max(sresid),length=40)
yfit<-dnorm(xfit)
lines(xfit,yfit)

#Non-constant error variance
#evaluat homoscedasticity
ncvTest(fit)
#plot studentized residuals vs. fitted values
spreadLevelPlot(fit)

#Multi-collinearity
vif(fit) #variance inflation factors
sqrt(vif(fit)) >2 

#Nonlinearity
#component + residual plot
crPlots(fit)
#Ceres plots
ceresPlots(fit)

#non-independence of errors
#Test for autocorrelation errors
durbinWatsonTest(fit)

#gvlma() function in gvlma package performs a global validataion
#of linear model assumptions as well separate evaluations of skewness 
#kurtosis and heteroscedasticity

#global test of model assumptions
library(gvlma)
gvmodel<-gvlma(fit)
summary(gvmodel)


###ANOVA
#Fit a model
#lower case letters are numeric variables and upper case letters are factors

#one way anova(completely randomized design)
fit<-aov(y~A, data=mydataframe)

#randomized block design (B is the blocking factor)
fit<-aov(y~A+B, data=mydataframe)

#two way factorial design
fit<- aov(y~A+B+A:B, data=mydataframe)
fit<-aov(y~A*B, data=mydataframe) 

#analysis of covariance
fit<-aov(y~A+x, data = mydataframe)

#one within Factor
fit<-aov(y~A+Error(subject/A), data=mydataframe)

#two within factors W1 W2, two between factors B1 B2
fit<-aov(y~(W1*W2*B1*B2) +Error(subject/(W1*W2)) +(B1+B2),data=mydataframe)

#diagnostic plots provide checks for heteroscedasticity
#normality and influential observerations
layout(matrix(c(1,2,3,4),2,2))
plot(fit)

#Evaluate model effects
#R provides Type I sequential SS 
#not the default Type III marginal SS reported by SAS and SPSS
#In a nonorthogonal design with more than one term on the right hand side
#of the equation order will matter 
#(i.e., A+B and B+A will produce different resuls)
#use drop1() to produce the Type III results
#it will compare each term with the full model
#alternatively we can use anova(fit.model1,fit.model2) to compare nested models

summary(fit) #display type I ANOVA table
drop1(fit, ~., test="F") #type III SS and F Tests

#multiple comparisons
#get Tukey HSD tests results based on Type I SS

#Tukey Honestly significant differences
TukeyHSD(fit) #fit comes from aov()

#two-way interaction plot
attach(mtcars)
gears<-factor(gears)
cyl<-factor(cyl)
interaction.plot(cy1,gear,mpg,type="b",col=c(1:3),
                 leg.bty="o",leg.bg = "beige", lwd=2,pch=c(18,24,22),
                 xlab="Number of Cylinders",
                 ylab = "Mean Miles Per Gallon",
                 main="Interaction Plot")

#Plot means with error bars
library(gplots)
attach(metcars)
cyl<-factor(cyl)
plotmeans(mpg~cyl,xlab="number of Cylinders",
          ylab="miles Per Gallon" , main="Mean Plot\nwith 95%CI" )

#If ther is more than one dependent(outcome) variable, 
#test them simultaneously using a multivariate analysis of variance (MANOVA)
#let Y be a matrix whose columns are the dependent variables

#2x2 factorial MANOVA with 3 dependent variables
Y<-cbind(y1,y2,y3)
fit<-manova(Y~A*B)
summary(fit,test="Pillai")

### Assessing Classical test assumptions
# inclassical parameteric procedures we often assume normality 
#and constant variance for the model error term

#Outliers aq.plot() function to identify multivariate outliers
#plotting the ordered squared robust Mahalanobis distances of the observations
#against the empirical distribution function of the MD^2i
#input consists of a matrix or data frame
#the function produces 4 graphs and returns 
#a boolean vector identifying the outliers

#detect outliers
library(mvoutlier)
outliers<- aq.plot(mtcars[c("mpg","disp","hp","drat","wt","qsec")])
outliers

#univeriate normality
#evaluate the normality of a variable using a Q-Q plot

#Q-Q plot for variable MPG
attach (mtcars)
qqnorm(mpg)
qqline(mpg)

#multivariate normality
#MANOVA assumes multivariate normality
#mshapiro.test() in package mvnormtest
#Shapiro-Wilk test for multivariate normality
#Input must be a numberic matrix

#test multivariate Normality
mshapiro.test(M)

#if we have px1 multivariate normal random vector X~N(mu,sigma)
#the squared Mahalanobis distance between x and mu 
#is going to be chi-square distributed with p degrees of freedom

#Graphical assessment of multivariate normality
x<-as.matrix(mydata) #n x p numeric matrix
center<- colMeans(x) #centroid
n<-nrow(x); p<-ncol(x); cov<-cov(x);
d<-mahalanobis(x,center,cov) #distances
qqplot(qchisq(ppoints(n),df=p),d,
       main="QQ plot assessing multivariate normality",
       ylab="Mahalanobis D2")
abline(a=0, b=1)

#Homogeneity of Variances
#bartlett.test() provides a parametric K-sample test of the equality of variances
#fligner.test() function provides a non-parametric test of the same
#y is  numeric variable G is grouping variable

#Bartlett Test of homogeneity of variances
bartlett.test(y~G, data=mydata)

#figner-killeen Test of homogeneity of variances
fligner.test(y~G, data=mydata)

#hovPlot() in HH package
library(HH)
hov(y~G,data=mydata)
hovPlot(y~G, data=mydata)


### Resampling statistics
#coin package perform a wide variety of re-randomization 
#or permutation based statistical tests
#these tests do no assume random sampling from well-defined populations
#lower case letters numerical variables
#upper case letters categorical factors
#Monte-Carlo simulation are availavle for all tests

#independent Two- and K-Sample location test

# Exact Wilcoxon Mann Whitney rank sum test
#y is numeric and A is a binary factor
library(cooin)
wilcox.test(y~A, data=mydata, distribution="exact")

#one-way permutation test based on 9999 monte-carlo resamplings
#y is numeric and A is a categorical factor
library(coin)
oneway_test(y~A, data=mydata, distribution=approximate(B=9999))

#independence of two numeric variables
#Spearman test of independence based on 9999 Monte-Carlo resamplings
#x, y are numeric variables
library(coin)
spearman_test(y~x, data=mydata, distribution=approximate(B=9999))

#independence in contingency tables
#Independence i 2-way contingency table based on 9999 Monte-Carlo resamplings
#A, B are factors
library(coin)
chisq.test(A~B, data=mydata, distribution=approximate (B=9999))

#3-way
mh_test(A~B|C, data=mydata, distribution=approximate (B=9999))

#linear by linear association test
lbl_test(A~B, data=mydata, distribution=approximate (B=9999))


###Power Analysis
#power analysis allows us to determine the sample size required
#to detect an effect of a given size with a given degree of confidence
#it allows us to determine the probability of detecting an effect of a given size
#with a given level of confidence under sample size constraints
#if the probability is unacceptably low we would alter or abandon the experiment

#below four quantities have an intimate relationship:
#1.sample size
#2.effect size
#3. significance level =P(Type I error)
#   = probability of finding an effect that is not there
#4. power=1-P(Type II error)
#   =probability of finding an effect that is there 

#given any three we can determine the fourth

#function	power calculations for
#pwr.2p.test	two proportions (equal n)
#pwr.2p2n.test	two proportions (unequal n)
#pwr.anova.test	balanced one way ANOVA
#pwr.chisq.test	chi-square test
#pwr.f2.test	general linear model
#pwr.p.test	proportion (one sample)
#pwr.r.test	correlation
#pwr.t.test	t-tests (one sample, 2 sample, paired)
#pwr.t2n.test	t-test (two samples with unequal n)

#t-tests
pwr.t.test(n = , d = , sig.level = ,
           power = , type = c("two.sample", "one.sample", "paired"))
#where n is the sample size, d is the effect size, and type indicates 
#a two-sample t-test, one-sample t-test or paired t-test

#unequal sample sizes, use
pwr.t2n.test(n1 = , n2= , d = , sig.level =, power = )

#ANOVA
#one-way analysis of variance
pwr.anova.test(k = , n = , f = , sig.level = , power = )
#k is the number of groups and n is the common sample size in each group

#Correkations
pwr.r.test(n = , r = , sig.level = , power = )

#lineaer Models
pwr.f2.test(u =, v = , f2 = , sig.level = , power = )

#test of proportions
pwr.2p.test(h = , n = , sig.level =, power = )

#unequal n's 
pwr.2p2n.test(h = , n1 = , n2 = , sig.level = , power = )

#single proportion
pwr.p.test(h = , n = , sig.level = power = )

#chi-square tests
pwr.chisq.test(w =, N = , df = , sig.level =, power = )

library(pwr)

# For a one-way ANOVA comparing 5 groups, calculate the
# sample size needed in each group to obtain a power of
# 0.80, when the effect size is moderate (0.25) and a
# significance level of 0.05 is employed.

pwr.anova.test(k=5,f=.25,sig.level=.05,power=.8)

# What is the power of a one-tailed t-test, with a
# significance level of 0.01, 25 people in each group,
# and an effect size equal to 0.75?

pwr.t.test(n=25,d=0.75,sig.level=.01,alternative="greater")

# Using a two-tailed test proportions, and assuming a
# significance level of 0.01 and a common sample size of
# 30 for each proportion, what effect size can be detected
# with a power of .75?

pwr.2p.test(n=30,sig.level=0.01,power=0.75)

# Plot sample size curves for detecting correlations of
# various sizes.

library(pwr)

# range of correlations
r <- seq(.1,.5,.01)
nr <- length(r)

# power values
p <- seq(.4,.9,.1)
np <- length(p)

# obtain sample sizes
samsize <- array(numeric(nr*np), dim=c(nr,np))
for (i in 1:np){
  for (j in 1:nr){
    result <- pwr.r.test(n = NULL, r = r[j],
                         sig.level = .05, power = p[i],
                         alternative = "two.sided")
    samsize[j,i] <- ceiling(result$n)
  }
}

# set up graph
xrange <- range(r)
yrange <- round(range(samsize))
colors <- rainbow(length(p))
plot(xrange, yrange, type="n",
     xlab="Correlation Coefficient (r)",
     ylab="Sample Size (n)" )

# add power curves
for (i in 1:np){
  lines(r, samsize[,i], type="l", lwd=2, col=colors[i])
}

# add annotation (grid lines, title, legend)
abline(v=0, h=seq(0,yrange[2],50), lty=2, col="grey89")
abline(h=0, v=seq(xrange[1],xrange[2],.02), lty=2,
       col="grey89")
title("Sample Size Estimation for Correlation Studies\n
  Sig=0.05 (Two-tailed)")
legend("topright", title="Power", as.character(p),
       fill=colors)

# with(data, expression)
# example applying a t-test to a data frame mydata
with(mydata, t.test(y ~ group))

# by(data, factorlist, function)
# example obtain variable means separately for
# each level of byvar in data frame mydata
by(mydata, mydata$byvar, function(x) mean(x))
