#include "dli.h"

struct subtract_tuple
{
    __host__ __device__ float operator()(const thrust::tuple<float, float> &t) const
    {
        return thrust::get<0>(t) - thrust::get<1>(t);
    }
};

void update_hx(int n, float dx, float dy, float dt,
               thrust::device_vector<float> &hx,
               thrust::device_vector<float> &ez)
{

    auto diff_it = thrust::make_transform_iterator(
        thrust::make_zip_iterator(thrust::make_tuple(ez.begin() + n, ez.begin())),
        subtract_tuple());

    thrust::transform(hx.begin(), hx.end() - n, diff_it, hx.begin(),
                      [dt, dx, dy] __host__ __device__(float h, float cex)
                      {
                          return h - dli::C0 * dt / 1.3f * cex / dy;
                      });
}

void update_hy(int n, float dx, float dy, float dt, thrust::device_vector<float> &hy,
               thrust::device_vector<float> &ez)
{

    auto diff_it = thrust::make_transform_iterator(
        thrust::make_zip_iterator(thrust::make_tuple(ez.begin(), ez.begin() + 1)),
        subtract_tuple());

    // thrust::transform(ez.begin(), ez.end() - 1, ez.begin() + 1, buffer.begin(),
    //                [] __host__ __device__ (float x, float y) { return x - y; });

    thrust::transform(hy.begin(), hy.end() - 1, diff_it, hy.begin(),
                      [dt, dx, dy] __host__ __device__(float h, float cey)
                      {
                          return h - dli::C0 * dt / 1.3f * cey / dx;
                      });
}

void update_dz(int n, float dx, float dy, float dt, thrust::device_vector<float> &hx_vec,
               thrust::device_vector<float> &hy_vec, thrust::device_vector<float> &dz_vec,
               int cells)
{

    float *hx = thrust::raw_pointer_cast(hx_vec.data());
    float *hy = thrust::raw_pointer_cast(hy_vec.data());
    float *dz = thrust::raw_pointer_cast(dz_vec.data());

    auto cell_ids_begin = thrust::make_counting_iterator(0);
    auto cell_ids_end = thrust::make_counting_iterator(cells);

    thrust::for_each(cell_ids_begin, cell_ids_end,
                     [n, dx, dy, dt, hx, hy, dz] __host__ __device__(int cell_id)
                     {
                         if (cell_id > n)
                         {
                             float hx_diff = hx[cell_id - n] - hx[cell_id];
                             float hy_diff = hy[cell_id] - hy[cell_id - 1];
                             dz[cell_id] += dli::C0 * dt * (hx_diff / dx + hy_diff / dy);
                         }
                     });
}

void update_ez(thrust::device_vector<float> &ez, thrust::device_vector<float> &dz)
{
    thrust::transform(dz.begin(), dz.end(), ez.begin(),
                      [] __host__ __device__(float d)
                      { return d / 1.3f; });
}

// Do not change the signature of this function
void simulate(int cells_along_dimension, float dx, float dy, float dt,
              thrust::device_vector<float> &d_hx,
              thrust::device_vector<float> &d_hy,
              thrust::device_vector<float> &d_dz,
              thrust::device_vector<float> &d_ez)
{

    int cells = cells_along_dimension * cells_along_dimension;

    for (int step = 0; step < dli::steps; step++)
    {
        update_hx(cells_along_dimension, dx, dy, dt, d_hx, d_ez);
        update_hy(cells_along_dimension, dx, dy, dt, d_hy, d_ez);
        update_dz(cells_along_dimension, dx, dy, dt, d_hx, d_hy, d_dz, cells);
        update_ez(d_ez, d_dz);
    }
}