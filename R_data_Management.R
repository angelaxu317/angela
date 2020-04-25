###Example 1
#purely random process with mean 0 and standard deviation 1
eps<- rnorm(100,mean=0,sd=1)
mu<-2 #the constant mean
#The process
X_t<-mu+eps

#plotting the time series
ts.plot(X_t, main="Example of random stationary time series")

###Creating new variables
#three examples for doing the same computations
mydata$sum <- mydata$x1 +mydata$x2
mydata$mean <- (mydata$x1 +mydata$x2)/2

attach(mydata)
mydata$sum <- x1+x2
mydata$mean <- (x1+x2)/2
detach(mydata)

mydata <- transform( mydata,
                     sum=x1+x2,
                     mean=(x1+x2)/2
                     )
###Recoding variables
#create 2 age categories
mydata$agecat <- ifelse(mydata$age >70, c("older"),c("younger"))

#create 3 age catefories
attach(mydata)
mydata$agecat[age>75] <- "Elder"
mydata$agecat[age>45 & age<=75] <- "Middle Aged"
mydata$agecat[age<45] <- "Young"
detach(mydata)

###Renaming variables
#rename interactively
fix(mydata) #results are saved on close

#rename programmatically
library(reshape)
mydata <- rename(mydata,c(oldname="newname"))

#you can re-enter all the variable names in order 
#changing the ones you need to change
#the limitation is that you need to enter all of them
names(mydata) <- c("x1","age","y","sex")


### Control Structures
#if-else
if (cond) expr
if (cond) expr1 else expr2
#for
for(var in seq) expr
#while
while(cond) expr
#switch
switch(expr,...)
#ifelse
ifelse(test,yes,no)

#example transpose of a matrix
#a poor alternative to built-in t() function

mytrans <- function(x) {
  if (!is.matrix(x)){
    warning("atgument is not a matrix: returning NA")
    return(NA_real_)
  }
  y <- matrix(1,nrow=ncol(x),ncol=nrow(x))
  for (i in 1:nrow(x)) {
    for (j in 1:ncol(x)){
      y[j,i] <- x[i,j]
    }
  }
  return(y)
}

#try it
z <- matrix(1:10, nrow=5,ncol=2)
tz <- mytrans(z)


###User-written Functions
myfunction <- function (arg1,arg2,...) {
  statements
  return(object)
}
#objects in the function are local to the function 
#the object returned can be any data type
#function example - get measures of central tendency
#and spread for a numberic vector x
#the user has a choice of measures and whether the results are printed
mysummary <- function(x,npar=TRUE,print=TRUE){
  if (!npar){
    center <- mean(x);spread <-sd(x)
  } else {
    center <-median(x); spread <-mad(x)
  }
  if (print & !npar) {
    cat("mean=", center, "\n","SD=",spread,"\n")
  }else if (print &npar) {
    cat("Median=",center,"\n","MAD=",spread,"\n")
  }
  result <- list (center=center,spread=spread)
  return (result)
}

#invoking the function
set.seed(1234)
x<- rpois(500,4)
y<-mysummary(x)
#Median= 4 
#MAD= 1.4826 

#y$center is the median(4)
#y$spread is the median absolute deviation (1.4826)

y<-mysummary(x,npar=FALSE, print=FALSE)
#no output
#y$center is the mean(4.052)
#y$spread is the standard deviation (2.01927)

###Sorting Data
#to sort data frame in R use the order() function. By default, soring is ASCENDING.
#prepend the sorting variable by a minus sign to indicate DECENDING order

#sorting examples using the mtcars dataset
attach(mtcars)
#sort by mpg
newdata <-mtcars[order(mpg),]
#sort by mpg and cy1
newdata<-mtcars[order(mpg,cy1),]
#sort by mpg(ascending) and cy1 (descending)
newdata<- mtcars[order(mpg,-cy1)]
detach(mtcars)

###Merging Data
#merge two data frames by ID
total<-merge(data frameA, data frameB, by='ID')

#merge two data frames by ID and country
total<-merge(data frameA, data frameB, by=c('ID',"Country"))

#adding rows
#to join two data frames(datasets) vertically use the rbind function
#the two data frames must have the same variables but they do not have to be in the same order
total<-rbind(data frameA, data frameB)


###Aggregating Data
#aggregate data fram mtcars by cyl and vs, returning means
#for numeric variables
attach(mtvars)
aggdata<-aggregate(mtcars, by=list(cyl,vs),
                   FUN=mean, na.rm=TRUE)
print(aggdata)
detach(mtcars)

#also summarize() in Hmisc package
#also summaryBy() in doBy package


###Reshaping Data
#Transpose use t() function to transpose a matrix or a data frame
#rownames become variable(column) names
#example using built-in dataset
mtcars
t(mtcars)

#eexample of melt function
library(reshape)
mdata<-melt(mydata, id=c("id","time"))

#cast the melted data
#cast(data, formula, function)
subjmeans<- cast(mdata,id~variable, mean)
timemeans<- cast(mdata, time~variable, mean)


###Subsetting Data
#R has powerful indexing features for accessing object elements
#selecting(keeping) variables

#select variables v1,v2,v3
myvar<-c("v1","v2","v3")
newdata<-mydata[myvars]

#another method
myvars<-paste("v",1:3, sep=" ")
newdata<-mydata[myvars]

#select 1st and 5th through 10th variables
newdata<-mydata[c(1,5:10)]

#excluding (dropping) variables
#exclude variables v1,v2,v3
myvars<- names(mydata) %in% c("v1","v2","v3")
newdata<-mydata[!myvars]

#exclude 3rd and 5th variable
newdata<-mydata[c(-3,-5)]

#delete variables v3, v5
mydata$v3<-mydata$v5 <-NULL

#select observations
#select first 5 observations
newdata<-mydata[1:5,]

#select baseed on variable values
newdata<-mydata[which (mydata$gender =="F"
                       & mydata$age>65),]

#or
attach(mydata)
newdata<-mydata[which(gender=="F" & age>65),]
detach(mydata)

#selection using the subset function

#using subset function
newdata<-subset(mydata,age>=20 | age <10, 
                select=c(ID,Weight))

newdata<-subset(mydata,sex=="m" &age>25,
                select=weight:income)
#random sample
#take a random sample of size 50 from a dataset mydata
#sample without replacement
mysample<-mydata[sample(1:nrow(mydata),50,
                        replace=FALSE),]













































