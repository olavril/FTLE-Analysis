#include <vector>
#include <limits>
#include "cstone/domain/domain.hpp"
#include "../../extern/Eigen/EigenValues"
#include "utils.hpp"
#include "find_neighbors.hpp"

using KeyType = uint64_t;
using Domain  = cstone::Domain<uint64_t, double, cstone::CpuTag>;

template<typename T>
inline T sinc(T x)
{
    T xpi = M_PI * x;
    if (x == 0) { return 1; }
    return std::sin(xpi) / xpi;
}

template<typename T>
inline T kernel_gradient(T dist, T rab, T h)
{
    const double K6 = 0.790450;

    T kern = K6 / std::pow(h, 3) * 6 * std::pow(sinc(dist / (2 * h)), 5) *
             (rab * std::cos(M_PI * dist / (2 * h)) / (dist * dist) -
              2 * h * rab * std::sin(M_PI * dist / (2 * h)) / (M_PI * std::pow(dist, 3)));

    return kern;
}

// return all the gradients for all neighbors
template<typename T>
std::tuple<std::vector<T>, std::vector<T>, std::vector<T>>
calculate_kernel_gradient(std::vector<T> distances, std::vector<T> rabx, std::vector<T> raby, std::vector<T> rabz,
                          std::vector<T> h)
{
    // kern has the same size as rab
    std::vector<T> kernx(distances.size(), 0.0);
    std::vector<T> kerny(distances.size(), 0.0);
    std::vector<T> kernz(distances.size(), 0.0);

#pragma omp parallel for
    for (size_t i = 0; i < distances.size(); i++)
    {
        if (distances[i] > 2 * h[i])
        {
            kernx[i] = 0;
            kerny[i] = 0;
            kernz[i] = 0;
        }
        else if (distances[i] != 0)
        {
            kernx[i] = kernel_gradient(distances[i], rabx[i], h[i]);
            kerny[i] = kernel_gradient(distances[i], raby[i], h[i]);
            kernz[i] = kernel_gradient(distances[i], rabz[i], h[i]);
        }
        else
        {
            kernx[i] = 0;
            kerny[i] = 0;
            kernz[i] = 0;
        }
    }

    return std::tie(kernx, kerny, kernz);
}

template<typename T>
T rab2_correction(T rab2)
{
    if (rab2 > 0.5) { rab2 -= 1.0; }
    else if (rab2 < -0.5) { rab2 += 1.0; }
    return rab2;
}

template<typename T>
std::vector<T> compute_FTLE(Domain domain, std::vector<T> x1, std::vector<T> y1, std::vector<T> z1, std::vector<T> x2,
                            std::vector<T> y2, std::vector<T> z2, std::vector<uint64_t> id, std::vector<T> h1,
                            std::vector<T> rho, T dt, size_t numParticles)
{
    /* argument list of python (LE,nneighbors,l, dt, pos1, pos2, V1, center_id, center_pos1, center_pos2, center_h1, h1,
     * mode)
     * nneighbors is the number of neighbors to consider (100-150)
     * pos1 is the positions of the first step, [numparticles+halo, 3]
     * pos2 is the positions of the second given step, [numparticles+halo, 3]
     * V1 is the volume of the first step, [numparticles]
     * h1 is the smoothing length of particles of the step, [numparticles+halo]
     *
     * center_id is the id of all the particles in the center, [numparticles]
     * center_pos1 is the positions of the centeral particles in the first step, [numparticles, 3]
     * center_pos2 is the positions of the centeral particles in the second step, [numparticles, 3]
     * center_h1 is the smoothing length of particles of the step, [numparticles]
     */

    // just particles, no halos
    // size_t startIndex = domain.startIndex();
    // size_t endIndex   = domain.endIndex();
    std::vector<T> FTLE(domain.nParticles(), 0.0);
    std::vector<T> V1(domain.nParticles(), 0.0);

    unsigned ng0   = 100;
    unsigned ngmax = 150;

    std::vector<cstone::LocalIndex> neighbors;
    std::vector<unsigned>           nc(domain.nParticles());

    resizeNeighbors(neighbors, domain.nParticles() * ngmax);
    findNeighborsSph(x1.data(), y1.data(), z1.data(), h1.data(), domain.startIndex(), domain.endIndex(), domain.box(),
                     domain.octreeProperties().nsView(), ng0, ngmax, neighbors.data(), nc.data() + domain.startIndex());

    for (size_t i = domain.startIndex(); i < domain.endIndex(); ++i)
    {
        V1[i] = 1.0 / (numParticles * rho[i]);
        // size_t particleID = id[i];

        // rab1, rab2 and dist sizes should be number of neighbors
        unsigned       neighborSize = nc[i];
        std::vector<T> rab1x(neighborSize);
        std::vector<T> rab1y(neighborSize);
        std::vector<T> rab1z(neighborSize);
        std::vector<T> dist1(neighborSize);

        std::vector<T> rab2x(neighborSize);
        std::vector<T> rab2y(neighborSize);
        std::vector<T> rab2z(neighborSize);
        std::vector<T> dist2(neighborSize);

        // TODO: update according to the neighbors.
        // Compute the distance between each particle and all the neighboring particles
        for (size_t j = 0; j < neighborSize; j++)
        {
            rab1x[j] = x1[i] - x1[neighbors[i * ngmax + j]];
            rab1y[j] = y1[i] - y1[neighbors[i * ngmax + j]];
            rab1z[j] = z1[i] - z1[neighbors[i * ngmax + j]];
            dist1[j] = std::sqrt(rab1x[j] * rab1x[j] + rab1y[j] * rab1y[j] + rab1z[j] * rab1z[j]);
        }

        // Compute the distance between each particle and all neighbors in the second step
        // If the corresponding particles in the second step has already been found, use it
        for (size_t j = 0; j < neighborSize; j++)
        {
            rab2x[j] = rab2_correction(x2[i] - x2[neighbors[i * ngmax + j]]);
            rab2y[j] = rab2_correction(y2[i] - y2[neighbors[i * ngmax + j]]);
            rab2z[j] = rab2_correction(z2[i] - z2[neighbors[i * ngmax + j]]);
        }

        // Sizes of dkV, dist1, rab1x, rab1y, rab1z, h1 should be the same and to the number of neighbors
        std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> dkV =
            calculate_kernel_gradient(dist1, rab1x, rab1y, rab1z, h1);
        std::vector<T> dkV0(rab1x.size());
        std::vector<T> dkV1(rab1x.size());
        std::vector<T> dkV2(rab1x.size());

        // Compute the gradient of the kernel for all the neighbors and multiply it
        // with the volume of the neighbor you calculate before. Loop over the neighbors
        for (size_t j = 0; j < neighborSize; j++)
        {
            dkV0[j] = std::get<0>(dkV)[j] * V1[neighbors[i * ngmax + j]];
            dkV1[j] = std::get<1>(dkV)[j] * V1[neighbors[i * ngmax + j]];
            dkV2[j] = std::get<2>(dkV)[j] * V1[neighbors[i * ngmax + j]];
        }

        Eigen::Matrix<T, 3, 3> Lmat;
        Eigen::Matrix<T, 3, 3> Fmat;
        Eigen::Matrix<T, 3, 3> Cmat;

        // Loop over the neighbors
        for (size_t j = 0; j < neighborSize; j++)
        {
            Lmat(0, 0) += rab1x[j] * dkV0[j];
            Lmat(0, 1) += rab1x[j] * dkV1[j];
            Lmat(0, 2) += rab1x[j] * dkV2[j];

            Lmat(1, 0) += rab1y[j] * dkV0[j];
            Lmat(1, 1) += rab1y[j] * dkV1[j];
            Lmat(1, 2) += rab1y[j] * dkV2[j];

            Lmat(2, 0) += rab1z[j] * dkV0[j];
            Lmat(2, 1) += rab1z[j] * dkV1[j];
            Lmat(2, 2) += rab1z[j] * dkV2[j];
        }
        Lmat = Eigen::Inverse(Lmat);

        std::vector<T> LdkV(3, 0.0);
        // Loop over the neighbors
        for (size_t j = 0; j < neighborSize; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                LdkV[k] += Lmat(k, 0) * dkV0[j] + Lmat(k, 1) * dkV1[j] + Lmat(k, 2) * dkV2[j];
            }

            Fmat(0, 0) += rab2x[j] * LdkV[0];
            Fmat(0, 1) += rab2x[j] * LdkV[1];
            Fmat(0, 2) += rab2x[j] * LdkV[2];

            Fmat(1, 0) += rab2y[j] * LdkV[0];
            Fmat(1, 1) += rab2y[j] * LdkV[1];
            Fmat(1, 2) += rab2y[j] * LdkV[2];

            Fmat(2, 0) += rab2z[j] * LdkV[0];
            Fmat(2, 1) += rab2z[j] * LdkV[1];
            Fmat(2, 2) += rab2z[j] * LdkV[2];
        }

        Cmat = Fmat.transpose() * Fmat;

        Eigen::EigenSolver<Eigen::Matrix<T, 3, 3>> eigensolver(Cmat);
        if (eigensolver.info() != Eigen::Success) abort();

        T maxEigenvalue = std::max({eigensolver.eigenvalues()[0].real(), eigensolver.eigenvalues()[1].real(),
                                    eigensolver.eigenvalues()[2].real()});

        FTLE[i] = std::log(std::sqrt(maxEigenvalue)) / dt;
    }

    return FTLE;
}
