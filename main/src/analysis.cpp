#include <algorithm>
#include "utils.hpp"
#include "arg_parser.hpp"
#include "ifile_io_impl.h"
#include "cstone/domain/domain.hpp"
#include "ftle.hpp"

using namespace sphexa;

void printAnalysisHelp(char* binName, int rank);
using Realtype = double;

int main(int argc, char** argv)
{
    auto [rank, numRanks] = initMpi();
    const ArgParser parser(argc, (const char**)argv);

    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printAnalysisHelp(argv[0], rank);
        return exitSuccess();
    }

    using Domain = cstone::Domain<uint64_t, double, cstone::CpuTag>;

    const std::string initFile1 = parser.get("--checkpoint1");
    const std::string initFile2 = parser.get("--checkpoint2");
    int               stepNo    = parser.get("--stepNo", 0);

    // read HDF5 checkpoint and decide which steps to process
    auto reader = makeH5PartReader(MPI_COMM_WORLD);
    reader->setStep(initFile1, stepNo, FileMode::collective);

    size_t numParticles  = reader->globalNumParticles(); // total number of particles in the simulation
    size_t simDimensions = std::cbrt(numParticles);      // dimension of the simulation

    std::vector<double>   x(reader->localNumParticles());
    std::vector<double>   y(reader->localNumParticles());
    std::vector<double>   z(reader->localNumParticles());
    std::vector<double>   h(reader->localNumParticles());
    std::vector<uint64_t> id(reader->localNumParticles());
    std::vector<double>   vx(reader->localNumParticles());
    std::vector<double>   vy(reader->localNumParticles());
    std::vector<double>   vz(reader->localNumParticles());
    std::vector<double>   rho(reader->localNumParticles());
    std::vector<double>   scratch1(x.size());
    std::vector<double>   scratch2(x.size());
    std::vector<double>   scratch3(x.size());

    reader->readField("x", x.data());
    reader->readField("y", y.data());
    reader->readField("z", z.data());
    reader->readField("h", h.data());
    reader->readField("rho", rho.data());
    reader->readField("id", id.data());
    reader->closeStep();

    std::cout << "Read " << reader->localNumParticles() << " particles on rank " << rank << std::endl;

    // create cornerstone tree
    std::vector<uint64_t> keys(x.size());
    size_t                bucketSizeFocus = 64;
    size_t                bucketSize      = std::max(bucketSizeFocus, numParticles / (100 * numRanks));
    float                 theta           = 1.0;
    cstone::Box<double>   box(-0.5, 0.5, cstone::BoundaryType::periodic); // boundary type from file?
    Domain                domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    domain.sync(keys, x, y, z, h, std::tie(vx, vy, vz), std::tie(scratch1, scratch2, scratch3));

    std::cout << "nparticles: " << domain.nParticles() << std::endl;
    std::cout << "nparticlesWithHalos: " << domain.nParticlesWithHalos() << std::endl;

    size_t first = domain.startIndex();
    size_t last  = domain.endIndex();
    // the difference between the physical times of the two steps
    double dt = 1.0;

    auto res = compute_FTLE(domain, x, y, z, x, y, z, id, h, rho, dt, numParticles);

    return exitSuccess();
}

void printAnalysisHelp(char* name, int rank)
{
    if (rank == 0)
    {
        printf("\nUsage:\n\n");
        printf("%s [OPTIONS]\n", name);
        printf("\nWhere possible options are:\n\n");

        printf("\t--checkpoint \t\t HDF5 checkpoint file with simulation data\n\n");
        printf("\t--stepNo \t\t Step number of the HDF5 checkpoint file with simulation data\n\n");
        printf("\t--meshSizeMultiplier \t\t Multiplier for the mesh size over the grid size.\n\n");
        printf("\t--numShells \t\t Number of shells for averaging. Default is half of mesh dimension read from the "
               "checkpoint data.\n\n");
    }
}