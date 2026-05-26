\name{selectCrossPlan}
\alias{selectCrossPlan}

\title{
Optimal Cross Plan Selection
}
\description{
Performs optimal cross plan selection using genetic algorithm optimization with additive and dominance effects.
}
\usage{
selectCrossPlan(
    cycleNumber, nCross, M, a, d, ploidy, targetAngle,
    probMut = 0.01, nMutate = 2L, nSel = 500L,
    nPop = 10000L, maxGen = 1000L, maxRun = 100L,
    anglePenalty = 0.5, cores = 4L
)
}
\arguments{
  \item{cycleNumber}{Cycle number of the breeding program.}
  \item{nCross}{Number of crosses to make.}
  \item{M}{Marker matrix (individuals × SNPs).}
  \item{a}{Vector of additive effects.}
  \item{d}{Vector of dominance effects.}
  \item{ploidy}{Ploidy level (e.g., 2, 4, 6).}
  \item{targetAngle}{Target angle in radians.}
  \item{probMut}{Mutation probability for progeny in GA.}
  \item{nMutate}{Number of potential mutations in mutated progeny.}
  \item{nSel}{Number of parents in GA.}
  \item{nPop}{Number of progeny in GA.}
  \item{maxGen}{Maximum number of generations.}
  \item{maxRun}{Stopping criteria for maximum number of runs without change.}
  \item{anglePenalty}{Penalty to vector length for off angle.}
  \item{cores}{Number of threads for parallel computation.}
}
\value{
A list containing:
\describe{
  \item{crossPlan}{Matrix of selected crosses with parent indices.}
  \item{uMax}{Maximum usefulness.}
  \item{simMax}{Maximum similarity.}
  \item{uBest}{Best usefulness value.}
  \item{simBest}{Best similarity value.}
  \item{angleBest}{Best angle value.}
}
}
\details{
This function extends optimal cross selection to include additive and dominance effects, supporting different ploidy levels. It uses a genetic algorithm to optimize crossing plans.
}
\references{
Woolliams, J. A., Berg, P., Dagnachew, B. S., & Meuwissen, T. H. E. (2015). Genetic contributions and their optimization. Journal of Animal Breeding and Genetics, 132(2), 89-99.
}
\examples{
# Example usage
nMarkers <- 100
nInds <- 10
M <- matrix(sample(0:2, nInds*nMarkers, replace=TRUE), nInds, nMarkers)
a <- rnorm(nMarkers)
d <- rnorm(nMarkers)
plan <- selectCrossPlan(cycleNumber=1, nCross=5, M=M, a=a, d=d,
                        ploidy=2, targetAngle=0.5)
head(plan$crossPlan)
}
