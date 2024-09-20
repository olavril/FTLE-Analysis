#include "gtest/gtest.h"
#include "ftle.hpp"
#include "ifile_io_impl.h"

// TEST(analysisTest, test1)
// {
//     int rank = 0, numRanks = 0;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
// }

TEST(analysisTest, test_kernel_gradient)
{
    std::vector<double> dist{0.72435803, 0.21992315, 0.92933338, 0.53326877};
    std::vector<double> h{0.44069504, 0.85591792, 0.07505942, 0.18832467};

    std::vector<double> rabx = {0.13440737, 0.38538617, 0.56836877, 0.51725883};

    std::vector<double> raby = {0.66748113, 0.78144886, 0.48422043, 0.87876388};

    std::vector<double> rabz = {0.72703719, 0.44812432, 0.51273011, 0.89805946};

    std::vector<double> expected_resultsx = {-5.49745160e-03, -2.80873062e+00, 0.00000000e+00, 0.00000000e+00};
    std::vector<double> expected_resultsy = {-2.73009228e-02, -5.69527268e+00, 0.00000000e+00, 0.00000000e+00};
    std::vector<double> expected_resultsz = {-2.97368501e-02, -3.26597213e+00, 0.00000000e+00, 0.00000000e+00};

    auto results  = calculate_kernel_gradient(dist, rabx, raby, rabz, h);
    auto resultsx = std::get<0>(results);
    auto resultsy = std::get<1>(results);
    auto resultsz = std::get<2>(results);

    for (size_t i = 0; i < resultsx.size(); i++)
    {
        EXPECT_NEAR(resultsx[i], expected_resultsx[i], 1e-6);
        EXPECT_NEAR(resultsy[i], expected_resultsy[i], 1e-6);
        EXPECT_NEAR(resultsz[i], expected_resultsz[i], 1e-6);
    }
}

TEST(analysisTest, test_inverse)
{
    Eigen::Matrix<double, 3, 3> mat;
    mat(0, 0) = 1;
    mat(0, 1) = 2;
    mat(0, 2) = -1;
    mat(1, 0) = 2;
    mat(1, 1) = 1;
    mat(1, 2) = 2;
    mat(2, 0) = -1;
    mat(2, 1) = 2;
    mat(2, 2) = 1;

    Eigen::Matrix<double, 3, 3> inv = Eigen::Inverse(mat);

    Eigen::Matrix<double, 3, 3> expected_inv;
    expected_inv(0, 0) = 3.0 / 16;
    expected_inv(0, 1) = 1.0 / 4;
    expected_inv(0, 2) = -5.0 / 16;
    expected_inv(1, 0) = 1.0 / 4;
    expected_inv(1, 1) = 0;
    expected_inv(1, 2) = 1.0 / 4;
    expected_inv(2, 0) = -5.0 / 16;
    expected_inv(2, 1) = 1.0 / 4;
    expected_inv(2, 2) = 3.0 / 16;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            EXPECT_NEAR(inv(i, j), expected_inv(i, j), 1e-6);
        }
    }
}

TEST(analysisTest, test_eigenvalues)
{
    Eigen::Matrix<double, 3, 3> mat;
    mat(0, 0) = 3;
    mat(0, 1) = 1;
    mat(0, 2) = 1;
    mat(1, 0) = 2;
    mat(1, 1) = 4;
    mat(1, 2) = 2;
    mat(2, 0) = 1;
    mat(2, 1) = 1;
    mat(2, 2) = 3;

    Eigen::EigenSolver<Eigen::Matrix<double, 3, 3>> eigensolver(mat);
    if (eigensolver.info() != Eigen::Success) abort();

    std::vector<double> expected_eigenvalues = {2, 6, 2};

    for (int i = 0; i < 3; i++)
    {
        EXPECT_NEAR(eigensolver.eigenvalues()[i].real(), expected_eigenvalues[i], 1e-6);
    }
}

TEST(analysisTest, test_domain_sync_order)
{
    auto [rank, numRanks] = initMpi();

    std::vector<double> x{0.4, -0.4, 0.4, -0.4};
    std::vector<double> y{0.4, -0.4, 0.4, -0.4};
    std::vector<double> z{0.4, 0.4, -0.4, -0.4};
    std::vector<double> h{1, 1, 1, 1};
    std::vector<double> id{4, 3, 2, 1};
    std::vector<double> vx{1.7, 1.8, 1.9, 2.0};
    std::vector<double> vy{2.1, 2.2, 2.3, 2.4};
    std::vector<double> vz{2.5, 2.6, 2.7, 2.8};
    std::vector<double> scratch1(x.size());
    std::vector<double> scratch2(x.size());
    std::vector<double> scratch3(x.size());

    std::vector<uint64_t> keys(x.size());
    size_t                bucketSizeFocus = 2;
    size_t                bucketSize      = 2;
    float                 theta           = 1.0;
    cstone::Box<double>   box(-0.5, 0.5, cstone::BoundaryType::periodic);
    Domain                domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    domain.sync(keys, x, y, z, h, std::tie(vx, id), std::tie(scratch1, scratch2, scratch3));

    if (rank == 0)
    {
        for (size_t i = 0; i < x.size(); i++)
        {
            std::cout << rank << " " << x[i] << " " << y[i] << " " << z[i] << " " << id[i] << std::endl;
        }
    }
    if (rank == 1)
    {
        for (size_t i = 0; i < x.size(); i++)
        {
            std::cout << rank << " " << x[i] << " " << y[i] << " " << z[i] << " " << id[i] << std::endl;
        }
    }
}

TEST(analysisTest, test_neighbor_finding)
{
    auto [rank, numRanks] = initMpi();
    std::vector<double> x{0.4, -0.4, 0.4, -0.4};
    std::vector<double> y{0.4, -0.4, 0.4, -0.4};
    std::vector<double> z{0.4, 0.4, -0.4, -0.4};
    std::vector<double> h{1, 1, 1, 1};
    std::vector<double> id{4, 3, 2, 1};
    std::vector<double> vx{1.7, 1.8, 1.9, 2.0};
    std::vector<double> vy{2.1, 2.2, 2.3, 2.4};
    std::vector<double> vz{2.5, 2.6, 2.7, 2.8};
    std::vector<double> scratch1(x.size());
    std::vector<double> scratch2(x.size());
    std::vector<double> scratch3(x.size());

    std::vector<uint64_t> keys(x.size());
    size_t                bucketSizeFocus = 2;
    size_t                bucketSize      = 2;
    float                 theta           = 1.0;
    cstone::Box<double>   box(-0.5, 0.5, cstone::BoundaryType::periodic);
    Domain                domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    domain.sync(keys, x, y, z, h, std::tie(vx, id), std::tie(scratch1, scratch2, scratch3));

    std::vector<cstone::LocalIndex> neighbors;
    std::vector<unsigned>           nc;

    resizeNeighbors(neighbors, domain.nParticles() * 150);
    findNeighborsSph(x.data(), y.data(), z.data(), h.data(), domain.startIndex(), domain.endIndex(), domain.box(),
                     domain.focusTree().treeLeaves(), 2, 2, neighbors.data(), nc.data() + domain.startIndex());

    if (rank == 0)
    {
        for (size_t i = 0; i < x.size(); i++)
        {
            std::cout << rank << " " << x[i] << " " << y[i] << " " << z[i] << " " << id[i] << " " << nc[i] << std::endl;
        }
    }
}