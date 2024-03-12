\name{selectCrosses}
\alias{selectCrosses}

\title{
Optimal Cross Selection
}
\description{
Performs optimal cross selection based on contribution theory as explained by Woolliams (2015) .
}
\usage{
selectCrosses(
    nCross, targetAngle, u, G, probMut = 0.01, 
    nMutate = 2L, nSel = 500L, 
    nPop = 10000L, maxGen = 1000L, maxRun = 100L, 
    anglePenalty = 0.5, nThreads = 4L
)
}

\arguments{
  \item{nCross}{
Number of crosses to make.
}
\item{targetAngle}{
Number of degrees to target from maximum usefulness (in radians).
}
\item{u}{
Vector of usefulness criterion for crosses.  
}
\item{G}{
Relationship matrix for individuals. 
}
\item{probMut}{
Mutation probability for progeny in GA. 
}
\item{nMutate}{
Number of potential mutations in mutated progeny.
}
\item{nSel}{
Number of parents in GA.
}
\item{nPop}{
Number of progeny in GA.
}
\item{maxGen}{
Maximum number of generations.
}
\item{maxRun}{
Stopping criteria for maximum number of runs without change.
}
\item{anglePenalty}{
Penalty to vector length for off angle, higher value emphasises angle more.
}
\item{nThreads}{
Number of threads for OpenMP.
}

}
\details{

Please refer to Kinghorn (1999) for technical details.

}
\value{
If arguments are properly specified the function returns:
\describe{
\item{result}{the same input object with additional rows in the predictions, metrics, modeling and status tables.}
}
}

\references{

Kinghorn, B. (1999). 19. Mate Selection for the tactical implementation of breeding programs. Proceedings of the Advancement of Animal Breeding and Genetics, 13, 130-133.

Woolliams, J. A., Berg, P., Dagnachew, B. S., & Meuwissen, T. H. E. (2015). Genetic contributions and their optimization. Journal of Animal Breeding and Genetics, 132(2), 89-99.

https://alphagenes.roslin.ed.ac.uk/wp/wp-content/uploads/2019/05/01_OptimalContributionSelection.pdf?x44213

}

\examples{
####=========================================####

#simulate population of 10 lines with 1000 markers
nMarkers=1000
nInds=10
M <- matrix(rep(0,nInds*nMarkers),nInds,nMarkers)
for (i in 1:nInds) {
  M[i,] <- ifelse(runif(nMarkers)<0.5,-1,1)
}
rownames(M) <- letters[1:10]
myrel <- rrBLUP::A.mat(M) # relationship matrix
mydata <- data.frame(predictedValue=rnorm(10), designation=letters[1:10]) # phenotypes

## define combinations and relationships
ebv <- data.frame(mydata[,c("predictedValue")]); rownames(ebv) <- mydata[,"designation"]
ebv <- data.frame(ebv[rownames(myrel),]); rownames(ebv) <- rownames(myrel)
crossComb = t(combn(1:nrow(myrel), 2)) # all possible cross combintations
eMP = (ebv[crossComb[,1],] +  ebv[crossComb[,2],])/2  # expected EBVs of all crosses based on # mean parent EBVs
K <- as.matrix(myrel)
# OCS: Determine a crossing plan
nCross <- 10
targetAngle <- 20


ebv <- data.frame(mydata[,c("predictedValue")]); rownames(ebv) <- mydata[,"designation"]
ebv <- data.frame(ebv[rownames(myrel),]); rownames(ebv) <- rownames(myrel)
crossComb = t(combn(1:nrow(myrel), 2)) # all possible cross combintations
eMP = (ebv[crossComb[,1],] +  ebv[crossComb[,2],])/2  # expected EBVs of all crosses based on # mean parent EBVs
K <- as.matrix(myrel)
# OCS: Determine a crossing plan
plan = cgiarOcs::selectCrosses(nCross=nCross, # number of crossed to be identified using OCS
                               targetAngle=((targetAngle)*pi)/180, # 30 degrees in radians
                               u=eMP, # expected cross mean EBVs
                               maxRun=100,
                               G=K)   # GRM

crossPlan <- as.data.frame(plan$crossPlan) # list of crosses to be made already sorted by best cross (lower the value the better the cross)
crossPlan[ ,1] <- rownames(K)[crossPlan[ ,1]] # name of parent 1
crossPlan[ ,2] <- rownames(K)[crossPlan[ ,2]] # name of parent 2
colnames(crossPlan) <- c("Parent1", "Parent2", "OCS.merit") # add names to table
eMPsel = (ebv[crossPlan[ ,1],] +     # expected EBVs of selected crosses based on
            ebv[crossPlan[ ,2],])/2  # mean parent EBVs
inbreeding = diag(K)
inbreedingSel = (inbreeding[crossPlan[ ,1]] + inbreeding[crossPlan[ ,2]])/2 # average inbreeding of the cross
treatment <- paste("~", paste(nCross,"crosses *",targetAngle, "degrees"))
result <- data.frame(module="ocs",analysisId="", 
                                          gid=1:nrow(crossPlan), designation=paste(crossPlan[,1],crossPlan[,2], sep=" x "),
                                          mother=crossPlan[,1],father=crossPlan[,2], entryType="predictedCross",
                                          treatment=treatment, breedingValueCross=eMPsel, inbreedingCross=inbreedingSel, ocsMerit=crossPlan[,3]
)
head(result)                                          
                     

}

