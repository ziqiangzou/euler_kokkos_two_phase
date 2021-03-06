
#pragma once

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "HydroBaseFunctor3D.h"
#include "shared/RiemannSolvers.h"

// init conditions
#include "shared/BlastParams.h"

namespace euler_kokkos { namespace all_regime
	{
		class ComputePressureCFunctor3D : public HydroBaseFunctor3D
		{
			public:
				ComputePressureCFunctor3D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor3D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& invDt, int nbCells)
				{
					ComputePressureCFunctor3D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, invDt);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& invDt) const
					{
					} // operator ()

				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (dst < src)
							dst = src;
					} // join

				const DataArrayConst Qdata;
		}; // ComputeTransportDtFunctor3D
		class ComputePressureRFunctor3D : public HydroBaseFunctor3D
		{
			public:
				ComputePressureRFunctor3D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor3D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& invDt, int nbCells)
				{
					ComputePressureRFunctor3D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, invDt);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& invDt) const
					{
					} // operator ()

				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (dst < src)
							dst = src;
					} // join

				const DataArrayConst Qdata;
		}; // ComputeTransportDtFunctor3D
		class ComputeVTProfileFunctor3D : public HydroBaseFunctor3D
		{
			public:
				ComputeVTProfileFunctor3D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor3D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, int& invDt, int nbCells)
				{
					ComputeVTProfileFunctor3D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, invDt);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, int& invDt) const
					{
					} // operator ()

				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (dst < src)
							dst = src;
					} // join

				const DataArrayConst Qdata;
		}; // ComputeTransportDtFunctor3D
		class ComputePressureFunctor3D : public HydroBaseFunctor3D
		{
			public:
				ComputePressureFunctor3D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor3D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& invDt, int nbCells)
				{
					ComputePressureFunctor3D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, invDt);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& invDt) const
					{
					} // operator ()

				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (dst < src)
							dst = src;
					} // join

				const DataArrayConst Qdata;
		}; // ComputeTransportDtFunctor3D
		class ComputeInterfaceRPositionFunctor3D : public HydroBaseFunctor3D
		{
			public:
				ComputeInterfaceRPositionFunctor3D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor3D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& invDt, int nbCells)
				{
					ComputeInterfaceRPositionFunctor3D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, invDt);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& invDt) const
					{
					} // operator ()

				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (dst < src)
							dst = src;
					} // join

				const DataArrayConst Qdata;
		}; // ComputeTransportDtFunctor3D
		class ComputeInterfacePositionFunctor3D : public HydroBaseFunctor3D
		{
			public:
				ComputeInterfacePositionFunctor3D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor3D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& invDt, int nbCells)
				{
					ComputeInterfacePositionFunctor3D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, invDt);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& invDt) const
					{
					} // operator ()

				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (dst < src)
							dst = src;
					} // join

				const DataArrayConst Qdata;
		}; // ComputeTransportDtFunctor3D
		class ComputeEnergyConservation2Functor3D : public HydroBaseFunctor3D
		{
			public:
				ComputeEnergyConservation2Functor3D(HydroParams params_, DataArrayConst Qdata_,DataArrayConst gradphi_) :
					HydroBaseFunctor3D(params_), Qdata(Qdata_),gradphi(gradphi_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata,DataArrayConst gradphi, real_t& invDt, int nbCells)
				{
					ComputeEnergyConservation2Functor3D functor(params, Qdata, gradphi);
					Kokkos::parallel_reduce(nbCells, functor, invDt);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& invDt) const
					{
					} // operator ()

				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (dst < src)
							dst = src;
					} // join

				const DataArrayConst Qdata;
				const DataArrayConst gradphi;
		}; // ComputeTransportDtFunctor3D
		class ComputeEnergyConservation1Functor3D : public HydroBaseFunctor3D
		{
			public:
				ComputeEnergyConservation1Functor3D(HydroParams params_, DataArrayConst Qdata_,DataArrayConst gradphi_) :
					HydroBaseFunctor3D(params_), Qdata(Qdata_),gradphi(gradphi_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata,DataArrayConst gradphi, real_t& invDt, int nbCells)
				{
					ComputeEnergyConservation1Functor3D functor(params, Qdata, gradphi);
					Kokkos::parallel_reduce(nbCells, functor, invDt);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& invDt) const
					{
					} // operator ()

				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (dst < src)
							dst = src;
					} // join

				const DataArrayConst Qdata;
				const DataArrayConst gradphi;
		}; // ComputeTransportDtFunctor3D
		class ComputeMassConservation1Functor3D : public HydroBaseFunctor3D
		{
			public:
				ComputeMassConservation1Functor3D(HydroParams params_, DataArrayConst Qdata_,DataArrayConst gradphi_) :
					HydroBaseFunctor3D(params_), Qdata(Qdata_),gradphi(gradphi_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata,DataArrayConst gradphi, real_t& invDt, int nbCells)
				{
					ComputeMassConservation1Functor3D functor(params, Qdata, gradphi);
					Kokkos::parallel_reduce(nbCells, functor, invDt);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& invDt) const
					{
					} // operator ()

				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (dst < src)
							dst = src;
					} // join

				const DataArrayConst Qdata;
				const DataArrayConst gradphi;
		}; // ComputeTransportDtFunctor3D
		class ComputeMassConservation2Functor3D : public HydroBaseFunctor3D
		{
			public:
				ComputeMassConservation2Functor3D(HydroParams params_, DataArrayConst Qdata_,DataArrayConst gradphi_) :
					HydroBaseFunctor3D(params_), Qdata(Qdata_),gradphi(gradphi_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata,DataArrayConst gradphi, real_t& invDt, int nbCells)
				{
					ComputeMassConservation2Functor3D functor(params, Qdata, gradphi);
					Kokkos::parallel_reduce(nbCells, functor, invDt);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& invDt) const
					{
					} // operator ()

				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (dst < src)
							dst = src;
					} // join

				const DataArrayConst Qdata;
				const DataArrayConst gradphi;
		}; // ComputeTransportDtFunctor3D
		class ComputeErrorGreshoFunctor3D : public HydroBaseFunctor3D
		{
			public:
				ComputeErrorGreshoFunctor3D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor3D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& invDt, int nbCells)
				{
					ComputeErrorGreshoFunctor3D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, invDt);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& invDt) const
					{
					} // operator ()

				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (dst < src)
							dst = src;
					} // join

				const DataArrayConst Qdata;
		}; // ComputeTransportDtFunctor3D
		class ComputeMaxVelocityFunctor3D : public HydroBaseFunctor3D
		{
			public:
				ComputeMaxVelocityFunctor3D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor3D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& invDt, int nbCells)
				{
					ComputeMaxVelocityFunctor3D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, invDt);
				}


				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						dst = ZERO_F;
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& invDt) const
					{
					} // operator ()

				// "Join" intermediate results from different threads.
				// This should normally implement the same reduction
				// operation as operator() above. Note that both input
				// arguments MUST be declared volatile.
				KOKKOS_INLINE_FUNCTION
					void join (volatile real_t& dst,
							const volatile real_t& src) const
					{
						// max reduce
						if (dst < src)
							dst = src;
					} // join

				const DataArrayConst Qdata;
		}; // ComputeTransportDtFunctor3D
}
}
