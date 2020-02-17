#pragma once

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#include <iostream>
#include <limits>
#include <iomanip>
#include <sstream>
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "HydroBaseFunctor2D.h"
#include "shared/RiemannSolvers.h"
#include "fstream"
#include<ctime>



namespace euler_kokkos { namespace all_regime
	{

		class ComputeAcousticStepFunctor2D : public HydroBaseFunctor2D
		{
			public:
				ComputeAcousticStepFunctor2D(HydroParams params_,
						DataArray Udata_, DataArrayConst Qdata_, DataArray gradphi_,
						real_t dt_) :
					HydroBaseFunctor2D(params_), Udata(Udata_), Qdata(Qdata_), gradphi(gradphi_), m_K(params.settings.K),
					dtdx(dt_/params.dx), dtdy(dt_/params.dy), half_dtdx(HALF_F * dtdx), half_dtdy(HALF_F * dtdy),
					conservative(params.settings.conservative) {};

				static void apply(HydroParams params,
						DataArray Udata, DataArrayConst Qdata, DataArray gradphi,
						real_t dt, int nbCells)
				{
					ComputeAcousticStepFunctor2D functor(params, Udata, Qdata, gradphi, dt);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void computeAcousticRelaxation(const HydroState& qLoc, real_t cLoc,
							const HydroState& qNei, real_t cNei,
							real_t M,  int IX, int dir,
							real_t& uStar, real_t& piStar) const
					{
						const real_t pLoc = computePressure(qLoc);
						const real_t pNei = computePressure(qNei);

						const real_t  aLoc = m_K * qLoc[ID] * FMAX(cLoc, cNei);
						const real_t  aNei = m_K * qNei[ID] * FMAX(cLoc, cNei);
						const real_t ratio = qLoc[ID]/(qLoc[ID] + qNei[ID]);

						uStar = dir * (qLoc[IX] * aLoc + qNei[IX] * aNei) / ( aLoc + aNei ) - (pNei - pLoc  + M) / (aLoc + aNei);

						const real_t theta = params.settings.low_mach_correction ? FMIN(abs(uStar) / FMAX(cNei, cLoc), ONE_F) : ONE_F;

						piStar = (aLoc * (pNei + (ONE_F - ratio) * M)  + aNei *  ( pLoc - ratio * M) )/(aLoc+ aNei)\
							 - dir  * aNei * aLoc * theta * (qNei[IX] - qLoc[IX])/( aNei + aLoc);


					}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						int i, j;
						index2coord(index, i, j, params.isize, params.jsize);

						const int ghostWidth = params.ghostWidth;

						if (j>=ghostWidth-1 && j<=params.jmax-ghostWidth+1 &&
								i>=ghostWidth-1 && i<=params.imax-ghostWidth+1)
						{
							const HydroState qLoc = getHydroState(Qdata, i, j);
							const real_t cLoc = computeSpeedSound(qLoc);
							const real_t psiLoc = psi(i, j);
							const real_t gradphixLoc = gradphi(i, j , IPX);
							const real_t gradphiyLoc = gradphi(i, j , IPY);
							const real_t KLoc = gradphi(i, j, IPK);
							const bool barotropic = qLoc[IH]>ZERO_F? params.settings.barotropic0 : params.settings.barotropic1;
							const real_t sigma = params.settings.sigma;

							real_t uStarMinusX, piStarMinusX, Mmx;
							{
								HydroState qMx = getHydroState(Qdata, i-1, j);

								const real_t cMinusX = computeSpeedSound(qMx);
								const real_t psiMx = psi(i-1, j);
								if (qMx[IH]*qLoc[IH]<=ZERO_F&&(fmax(qMx[IH], qLoc[IH])>ZERO_F))
								{
									int delta_i=-1, delta_j=0;
									const real_t gradphixMx = gradphi(i-1, j , IPX);
									const real_t gradphiyMx = gradphi(i-1, j , IPY);
									real_t dphix, dphiy;
									computeGradphiInter(gradphixLoc, gradphiyLoc, qLoc[IH], gradphixMx, gradphiyMx, qMx[IH], dphix, dphiy);
									real_t deltap = ZERO_F;
									if (sigma > ZERO_F)
									{
										const real_t KMx = gradphi(i-1, j, IPK);
										computeCurvatureInter(KLoc, KMx, qLoc[IH], qMx[IH], deltap);
									}
									const real_t deltapsi = psiMx - psiLoc;
									computeAcousticRelaxationInter(qLoc, qMx, dphix, dphiy, delta_i, delta_j, deltap, deltapsi,  uStarMinusX, piStarMinusX);

									Mmx=qLoc[ID]* deltapsi;

								}
								else
								{
									Mmx = computeM(qLoc, psiLoc, qMx, psiMx);
									computeAcousticRelaxation(qLoc, cLoc, qMx, cMinusX, Mmx,  IU, -1,
											uStarMinusX, piStarMinusX);
								}
							}

							real_t uStarPlusX, piStarPlusX, Mpx;
							{
								HydroState qPx = getHydroState(Qdata, i+1, j);
								const real_t cPlusX = computeSpeedSound(qPx);
								const real_t psiPx = psi(i+1, j);
								if (qPx[IH]*qLoc[IH]<=ZERO_F&&(fmax(qPx[IH], qLoc[IH])>ZERO_F))
								{
									int delta_i=1, delta_j=0;
									const real_t gradphixPx = gradphi(i+1, j , IPX);
									const real_t gradphiyPx = gradphi(i+1, j , IPY);
									real_t dphix, dphiy;
									computeGradphiInter(gradphixLoc, gradphiyLoc, qLoc[IH], gradphixPx, gradphiyPx, qPx[IH], dphix, dphiy);
									real_t deltap = ZERO_F;
									if (sigma > ZERO_F)
									{
										const real_t KPx = gradphi(i+1, j, IPK);
										computeCurvatureInter(KLoc, KPx, qLoc[IH], qPx[IH], deltap);
									}
									const real_t deltapsi = psiPx - psiLoc;
									computeAcousticRelaxationInter(qLoc, qPx, dphix, dphiy, delta_i, delta_j, deltap, deltapsi,  uStarPlusX, piStarPlusX);
									Mpx=qLoc[ID]* deltapsi;
								}
								else
								{
									Mpx = computeM(qLoc, psiLoc, qPx, psiPx);
									computeAcousticRelaxation(qLoc, cLoc, qPx, cPlusX, Mpx, IU, +1,
											uStarPlusX, piStarPlusX);
								}
							}

							real_t uStarMinusY, piStarMinusY, Mmy;
							{
								HydroState qMy = getHydroState(Qdata, i, j-1);
								const real_t cMinusY = computeSpeedSound(qMy);
								const real_t psiMy = psi(i, j-1);
								if (qMy[IH]*qLoc[IH]<=ZERO_F&&(fmax(qMy[IH], qLoc[IH])>ZERO_F))
								{
									int delta_i=0, delta_j=-1;
									const real_t gradphixMy = gradphi(i, j-1 , IPX);
									const real_t gradphiyMy = gradphi(i, j-1 , IPY);
									real_t dphix, dphiy;
									computeGradphiInter(gradphixLoc, gradphiyLoc, qLoc[IH], gradphixMy, gradphiyMy, qMy[IH], dphix, dphiy);
									real_t deltap = ZERO_F;
									if (sigma > ZERO_F)
									{
										const real_t KMy = gradphi(i, j-1, IPK);
										computeCurvatureInter(KLoc, KMy, qLoc[IH], qMy[IH], deltap);
									}
									const real_t deltapsi = psiMy - psiLoc;
									computeAcousticRelaxationInter(qLoc, qMy, dphix, dphiy, delta_i, delta_j, deltap, deltapsi,  uStarMinusY, piStarMinusY);
									Mmy=qLoc[ID]* deltapsi;
								}
								else
								{
									Mmy = computeM(qLoc, psiLoc, qMy, psiMy);
									computeAcousticRelaxation(qLoc, cLoc, qMy, cMinusY, Mmy, IV, -1,
											uStarMinusY, piStarMinusY);
								}
							}

							real_t uStarPlusY, piStarPlusY, Mpy;
							{
								HydroState qPy = getHydroState(Qdata, i, j+1);
								const real_t cPlusY = computeSpeedSound(qPy);
								const real_t psiPy = psi(i, j+1);
								if (qPy[IH]*qLoc[IH]<=ZERO_F&&(fmax(qPy[IH], qLoc[IH])>ZERO_F))
								{
									int delta_i=0, delta_j= 1;
									const real_t gradphixPy = gradphi(i, j+1 , IPX);
									const real_t gradphiyPy = gradphi(i, j+1 , IPY);
									real_t dphix, dphiy;
									computeGradphiInter(gradphixLoc, gradphiyLoc, qLoc[IH], gradphixPy, gradphiyPy, qPy[IH], dphix, dphiy);
									real_t deltap = ZERO_F;
									if (sigma > ZERO_F)
									{
										const real_t KPy = gradphi(i, j+1, IPK);
										computeCurvatureInter(KLoc, KPy, qLoc[IH], qPy[IH], deltap);
									}
									const real_t deltapsi = psiPy - psiLoc;
									computeAcousticRelaxationInter(qLoc, qPy, dphix, dphiy, delta_i, delta_j, deltap, deltapsi,  uStarPlusY, piStarPlusY);
									Mpy=qLoc[ID]* deltapsi;
								}
								else
								{
									Mpy = computeM(qLoc, psiLoc, qPy, psiPy);
									computeAcousticRelaxation(qLoc, cLoc, qPy, cPlusY, Mpy,  IV, +1,
											uStarPlusY, piStarPlusY);
								}
							}

							HydroState uLoc = getHydroState(Udata, i, j);

							// Acoustic update
							uLoc[IU] -= dtdx * (piStarPlusX - piStarMinusX);
							uLoc[IV] -= dtdy * (piStarPlusY - piStarMinusY);
							uLoc[IU] -= half_dtdx * (Mpx - Mmx);
							uLoc[IV] -= half_dtdy * (Mpy - Mmy);

							if (!barotropic)
							{
								uLoc[IP] -= dtdx * (piStarMinusX * uStarMinusX + piStarPlusX * uStarPlusX);
								uLoc[IP] -= dtdy * (piStarMinusY * uStarMinusY + piStarPlusY * uStarPlusY);
								uLoc[IP] -= half_dtdx * (Mpx * uStarPlusX + Mmx * uStarMinusX);
								uLoc[IP] -= half_dtdy * (Mpy * uStarPlusY + Mmy * uStarMinusY);
							}



							// Compute L factor
							const real_t L = (ONE_F+
									dtdx * (uStarMinusX + uStarPlusX)+
									dtdy * (uStarMinusY + uStarPlusY));
							const real_t invL = ONE_F / L;

							uLoc[ID] *= invL;
							uLoc[IU] *= invL;
							uLoc[IV] *= invL;
							if (!barotropic)
								uLoc[IP] *= invL;


							// Real update
							setHydroState(Udata, uLoc, i, j);
						}
					}

				const DataArray Udata;
				const DataArrayConst Qdata;
				const DataArray gradphi;
				const real_t m_K;
				const real_t dtdx;
				const real_t dtdy;
				const real_t half_dtdx;
				const real_t half_dtdy;
				const bool conservative;
		}; // ComputeAcousticStepFunctor2D


		class ComputeTransportStepFunctor2D : HydroBaseFunctor2D
		{
			public:
				ComputeTransportStepFunctor2D(HydroParams params_,
						DataArrayConst Udata_, DataArrayConst Qdata_, DataArray U2data_, DataArray gradphi_,
						real_t dt_) :
					HydroBaseFunctor2D(params_),
					Udata(Udata_), Qdata(Qdata_), U2data(U2data_), gradphi(gradphi_),
					dtdx(dt_/params.dx), dtdy(dt_/params.dy),
					conservative(params.settings.conservative) {};

				static void apply(HydroParams params,
						DataArrayConst Udata, DataArrayConst Qdata, DataArray U2data, DataArray gradphi,
						real_t dt, int nbCells)
				{
					ComputeTransportStepFunctor2D functor(params, Udata, Qdata, U2data, gradphi, dt);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void computeAcousticRelaxation(const HydroState & qLoc, real_t cLoc,
							const HydroState & qNei, real_t cNei,
							real_t M,  int IX, int dir, real_t& uStar) const
					{
						const real_t pLoc = computePressure(qLoc);
						const real_t pNei = computePressure(qNei);

						const real_t  aLoc = params.settings.K * qLoc[ID] * FMAX(cLoc, cNei);
						const real_t  aNei = params.settings.K * qNei[ID] * FMAX(cLoc, cNei);
						uStar = dir * (qLoc[IX] * aLoc + qNei[IX] * aNei) / ( aLoc + aNei ) - (pNei - pLoc  + M) / (aLoc + aNei);
					}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);

						const int ghostWidth = params.ghostWidth;

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const HydroState qLoc = getHydroState(Qdata, i, j);
							const real_t cLoc = computeSpeedSound(qLoc);
							const real_t psiLoc = psi(i, j);
							const real_t KLoc = gradphi(i, j, IPK);
							const real_t gradphixLoc = gradphi(i, j , IPX);
							const real_t gradphiyLoc = gradphi(i, j , IPY);
							const real_t sigma = params.settings.sigma;


							real_t uStarMinusX;
							{
								HydroState qMx = getHydroState(Qdata, i-1, j);
								const real_t cMinusX = computeSpeedSound(qMx);
								const real_t psiMx = psi(i-1, j);
								if (qMx[IH]*qLoc[IH]<=ZERO_F&&(fmax(qMx[IH], qLoc[IH])>ZERO_F))
								{
									int delta_i=-1, delta_j=0;
									const real_t gradphixMx = gradphi(i-1, j , IPX);
									const real_t gradphiyMx = gradphi(i-1, j , IPY);
									real_t dphix, dphiy;
									computeGradphiInter(gradphixLoc, gradphiyLoc, qLoc[IH], gradphixMx, gradphiyMx, qMx[IH], dphix, dphiy);
									real_t deltap = ZERO_F;
									if (sigma > ZERO_F)
									{
										const real_t KMx = gradphi(i-1, j, IPK);
										computeCurvatureInter(KLoc, KMx, qLoc[IH], qMx[IH], deltap);
									}
									const real_t deltapsi = psiMx - psiLoc;
									real_t piStarMinusX;
									computeAcousticRelaxationInter(qLoc, qMx, dphix, dphiy, delta_i, delta_j, deltap, deltapsi,  uStarMinusX, piStarMinusX);
								}
								else
								{
									const real_t Mmx = computeM(qLoc, psiLoc, qMx, psiMx);
									computeAcousticRelaxation(qLoc, cLoc, qMx, cMinusX, Mmx, IU, -1,
											uStarMinusX);
								}
							}

							real_t uStarPlusX;
							{
								HydroState qPx = getHydroState(Qdata, i+1, j);
								const real_t cPlusX = computeSpeedSound(qPx);
								const real_t psiPx = psi(i+1, j);
								if (qPx[IH]*qLoc[IH]<=ZERO_F&&(fmax(qPx[IH], qLoc[IH])>ZERO_F))
								{
									int delta_i=1, delta_j=0;
									const real_t gradphixPx = gradphi(i+1, j , IPX);
									const real_t gradphiyPx = gradphi(i+1, j , IPY);
									real_t dphix, dphiy;
									computeGradphiInter(gradphixLoc, gradphiyLoc, qLoc[IH], gradphixPx, gradphiyPx, qPx[IH], dphix, dphiy);
									real_t deltap = ZERO_F;
									if (sigma > ZERO_F)
									{
										const real_t KPx = gradphi(i+1, j, IPK);
										computeCurvatureInter(KLoc, KPx, qLoc[IH], qPx[IH], deltap);
									}
									const real_t deltapsi = psiPx - psiLoc;
									real_t piStarPlusX;
									computeAcousticRelaxationInter(qLoc, qPx, dphix, dphiy, delta_i, delta_j, deltap, deltapsi,  uStarPlusX, piStarPlusX);
								}
								else
								{
									const real_t Mpx = computeM(qLoc, psiLoc, qPx, psiPx);
									computeAcousticRelaxation(qLoc, cLoc, qPx, cPlusX, Mpx,IU, +1,
											uStarPlusX);
								}
							}

							real_t uStarMinusY;
							{
								HydroState qMy = getHydroState(Qdata, i, j-1);
								const real_t cMinusY = computeSpeedSound(qMy);
								const real_t psiMy = psi(i, j-1);
								if (qMy[IH]*qLoc[IH]<=ZERO_F&&(fmax(qMy[IH], qLoc[IH])>ZERO_F))
								{
									int delta_i=0, delta_j=-1;
									const real_t gradphixMy = gradphi(i, j-1 , IPX);
									const real_t gradphiyMy = gradphi(i, j-1 , IPY);
									real_t dphix, dphiy;
									computeGradphiInter(gradphixLoc, gradphiyLoc, qLoc[IH], gradphixMy, gradphiyMy, qMy[IH], dphix, dphiy);
									real_t deltap = ZERO_F;
									if (sigma > ZERO_F)
									{
										const real_t KMy = gradphi(i, j-1, IPK);
										computeCurvatureInter(KLoc, KMy, qLoc[IH], qMy[IH], deltap);
									}
									const real_t deltapsi = psiMy - psiLoc;
									real_t piStarMinusY;
									computeAcousticRelaxationInter(qLoc, qMy, dphix, dphiy, delta_i, delta_j, deltap, deltapsi,  uStarMinusY, piStarMinusY);
								}
								else
								{
									const real_t Mmy = computeM(qLoc, psiLoc, qMy, psiMy);
									computeAcousticRelaxation(qLoc, cLoc, qMy, cMinusY, Mmy, IV, -1,
											uStarMinusY);
								}
							}

							real_t uStarPlusY;
							{
								HydroState qPy = getHydroState(Qdata, i, j+1);
								const real_t cPlusY = computeSpeedSound(qPy);
								const real_t psiPy = psi(i, j+1);
								if (qPy[IH]*qLoc[IH]<=ZERO_F&&(fmax(qPy[IH], qLoc[IH])>ZERO_F))
								{
									int delta_i=0, delta_j= 1;
									const real_t gradphixPy = gradphi(i, j+1 , IPX);
									const real_t gradphiyPy = gradphi(i, j+1 , IPY);
									real_t dphix, dphiy;
									computeGradphiInter(gradphixLoc, gradphiyLoc, qLoc[IH], gradphixPy, gradphiyPy, qPy[IH], dphix, dphiy);
									real_t deltap = ZERO_F;
									if (sigma > ZERO_F)
									{
										const real_t KPy = gradphi(i, j+1, IPK);
										computeCurvatureInter(KLoc, KPy, qLoc[IH], qPy[IH], deltap);
									}
									const real_t deltapsi = psiPy - psiLoc;
									real_t piStarPlusY;
									computeAcousticRelaxationInter(qLoc, qPy, dphix, dphiy, delta_i, delta_j, deltap, deltapsi,  uStarPlusY, piStarPlusY);
								}
								else
								{
									const real_t Mpy = computeM(qLoc, psiLoc, qPy, psiPy);
									computeAcousticRelaxation(qLoc, cLoc, qPy, cPlusY, Mpy, IV, +1,
											uStarPlusY);
								}
							}

							const HydroState uLoc = getHydroState(Udata, i, j);
							HydroState u2Loc = getHydroState(Udata, i, j);

							u2Loc[ID] += dtdx * uLoc[ID] * (uStarMinusX + uStarPlusX);
							u2Loc[IU] += dtdx * uLoc[IU] * (uStarMinusX + uStarPlusX);
							u2Loc[IV] += dtdx * uLoc[IV] * (uStarMinusX + uStarPlusX);
							u2Loc[IP] += dtdx * uLoc[IP] * (uStarMinusX + uStarPlusX);

							u2Loc[ID] += dtdy * uLoc[ID] * (uStarMinusY + uStarPlusY);
							u2Loc[IU] += dtdy * uLoc[IU] * (uStarMinusY + uStarPlusY);
							u2Loc[IV] += dtdy * uLoc[IV] * (uStarMinusY + uStarPlusY);
							u2Loc[IP] += dtdy * uLoc[IP] * (uStarMinusY + uStarPlusY);

							{
								const int i0 = (uStarMinusX > ZERO_F) ? i : i - 1;
								HydroState u0 = getHydroState(Udata, i0, j);
								if (qLoc[IH]*u0[IH]<=ZERO_F&& (qLoc[IH]>ZERO_F||u0[IH]>ZERO_F))
								{
									HydroState uMx = getHydroState(Udata, i0-1, j);
									HydroState uMy = getHydroState(Udata, i0, j-1);
									HydroState uPy = getHydroState(Udata, i0, j+1);
									const real_t dphix = gradphi(i0, j, IPX);
									const real_t dphiy = gradphi(i0, j, IPY);

									u0= compute_GhostState(i0, j, u0, uMx, uLoc, uMy, uPy, dphix, dphiy);
								}
								u2Loc[ID] -= dtdx * u0[ID] * uStarMinusX;
								u2Loc[IU] -= dtdx * u0[IU] * uStarMinusX;
								u2Loc[IV] -= dtdx * u0[IV] * uStarMinusX;
								u2Loc[IP] -= dtdx * u0[IP] * uStarMinusX;
							}

							{
								const int i0= (uStarPlusX > ZERO_F) ? i : i + 1;
								HydroState u0 = getHydroState(Udata, i0, j);
								if (qLoc[IH]*u0[IH]<=ZERO_F&& (qLoc[IH]>ZERO_F||u0[IH]>ZERO_F))
								{
									HydroState uPx = getHydroState(Udata, i0+1, j);
									HydroState uMy = getHydroState(Udata, i0, j-1);
									HydroState uPy = getHydroState(Udata, i0, j+1);

									const real_t dphix = gradphi(i0, j, IPX);
									const real_t dphiy = gradphi(i0, j, IPY);

									u0= compute_GhostState(i0, j, u0, uLoc, uPx, uMy, uPy, dphix, dphiy);

								}
								u2Loc[ID] -= dtdx * u0[ID] * uStarPlusX;
								u2Loc[IU] -= dtdx * u0[IU] * uStarPlusX;
								u2Loc[IV] -= dtdx * u0[IV] * uStarPlusX;
								u2Loc[IP] -= dtdx * u0[IP] * uStarPlusX;
							}

							{
								const int j0 = (uStarMinusY > ZERO_F) ? j : j - 1;
								HydroState u0 = getHydroState(Udata, i, j0);
								if (qLoc[IH]*u0[IH]<=ZERO_F&& (qLoc[IH]>ZERO_F||u0[IH]>ZERO_F))
								{
									HydroState uMy = getHydroState(Udata, i, j0-1);
									HydroState uMx = getHydroState(Udata, i-1, j0);
									HydroState uPx = getHydroState(Udata, i+1, j0);

									const real_t dphix = gradphi(i, j0, IPX);
									const real_t dphiy = gradphi(i, j0, IPY);

									u0= compute_GhostState(i, j0, u0, uMx, uPx, uMy, uLoc, dphix, dphiy);

								}
								u2Loc[ID] -= dtdy * u0[ID] * uStarMinusY;
								u2Loc[IU] -= dtdy * u0[IU] * uStarMinusY;
								u2Loc[IV] -= dtdy * u0[IV] * uStarMinusY;
								u2Loc[IP] -= dtdy * u0[IP] * uStarMinusY;
							}

							{
								const int j0 = (uStarPlusY > ZERO_F) ? j : j + 1;
								HydroState u0 = getHydroState(Udata, i, j0);
								if (qLoc[IH]*u0[IH]<=ZERO_F&& (qLoc[IH]>ZERO_F||u0[IH]>ZERO_F))
								{
									HydroState uPy = getHydroState(Udata, i, j0+1);
									HydroState uMx = getHydroState(Udata, i-1, j0);
									HydroState uPx = getHydroState(Udata, i+1, j0);

									const real_t dphix = gradphi(i, j0, IPX);
									const real_t dphiy = gradphi(i, j0, IPY);

									u0= compute_GhostState(i, j0, u0, uMx, uPx, uLoc, uPy, dphix, dphiy);
								}
								u2Loc[ID] -= dtdy * u0[ID] * uStarPlusY;
								u2Loc[IU] -= dtdy * u0[IU] * uStarPlusY;
								u2Loc[IV] -= dtdy * u0[IV] * uStarPlusY;
								u2Loc[IP] -= dtdy * u0[IP] * uStarPlusY;
							}
							setHydroState(U2data, u2Loc, i, j);


						}
					}

				const DataArrayConst Udata;
				const DataArrayConst Qdata;
				const DataArray U2data;
				const DataArray gradphi;
				const real_t dtdx;
				const real_t dtdy;
				const bool conservative;
		}; // ComputeTransportStepFunctor2D

		class ComputeStateChangeFunctor2D : HydroBaseFunctor2D
		{
			public:
				ComputeStateChangeFunctor2D(HydroParams params_, DataArray Udata_, DataArray Qdata_, DataArray gradphi_):
					HydroBaseFunctor2D(params_), Udata(Udata_),Qdata(Qdata_), gradphi(gradphi_){};
				static void apply(HydroParams params,
						DataArray Udata, DataArray Qdata, DataArray gradphi,
						int nbCells)
				{
					ComputeStateChangeFunctor2D functor(params, Udata, Qdata, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}
				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						const int ghostWidth = params.ghostWidth;
						int i, j;
						index2coord(index, i, j, params.isize, params.jsize);

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							HydroState qLoc=getHydroState(Qdata, i, j);
							real_t phi =Udata(i, j, IH);
							real_t phi0=qLoc[IH];
							real_t gradphix=gradphi(i, j, IPX);
							real_t gradphiy=gradphi(i, j, IPY);
							const real_t gradphi_mod = sqrt(gradphix*gradphix+gradphiy*gradphiy);
							gradphix/=gradphi_mod;
							gradphiy/=gradphi_mod;
							const bool   barotropic = phi >ZERO_F? params.settings.barotropic0 : params.settings.barotropic1;
							const bool   barotropic0= phi0>ZERO_F? params.settings.barotropic0 : params.settings.barotropic1;

							if (phi*phi0<=ZERO_F&&(phi0>ZERO_F||phi>ZERO_F))
							{
								real_t number=0.;
								real_t p=ZERO_F, rho=ZERO_F, u=ZERO_F, v=ZERO_F;
								const real_t g_x=params.settings.g_x;
								const real_t g_y=params.settings.g_y;
								const real_t dx=params.dx;
								const real_t dy=params.dy;
								if (i>ghostWidth)
								{
									const real_t phiMx=Qdata(i-1, j, IH);
									if (fmin(phiMx, phi)>ZERO_F||fmax(phiMx, phi)<=ZERO_F)
									{
										HydroState qMx=getHydroState(Qdata, i-1, j);

										real_t rho0=qMx[ID], p0=qMx[IP];
										real_t delta_i=-1;
										const real_t num=ONE_F;
										const real_t vn    =qMx[IU]*gradphix    +qMx[IV]*gradphiy;
										const real_t vt    =qMx[IV]*gradphix    -qMx[IU]*gradphiy;
										const real_t dp=g_x*dx*delta_i;

										if (!barotropic)
										{
											p0-=dp*rho0;
										}
										else
										{
											const real_t pMx = computePressure(qMx);
											qMx[IP] =pMx - dp * rho0;
											rho0 = computeDensity(qMx);
										}

										rho+=rho0*num;
										p+=p0*num;
										u+=vn*num;
										v+=vt*num;
										number+=num;
									}
								}
								if (i<params.imax-ghostWidth)
								{
									const real_t phiPx=Qdata(i+1, j, IH);
									if (fmin(phiPx, phi)>ZERO_F||fmax(phiPx, phi)<=ZERO_F)
									{
										HydroState qPx=getHydroState(Qdata, i+1, j);

										real_t rho0=qPx[ID], p0=qPx[IP];
										real_t delta_i=1;
										const real_t vn    =qPx[IU]*gradphix    +qPx[IV]*gradphiy;
										const real_t vt    =qPx[IV]*gradphix    -qPx[IU]*gradphiy;
										const real_t num=ONE_F;
										const real_t dp=g_x*dx*delta_i;
										if (!barotropic)
										{
											p0-=dp*rho0;
										}
										else
										{
											const real_t pPx = computePressure(qPx);
											qPx[IP] =pPx - dp * rho0;
											rho0 = computeDensity(qPx);
										}

										rho+=rho0*num;
										p+=p0*num;
										u+=vn*num;
										v+=vt*num;
										number+=num;
									}
								}
								if (j>ghostWidth)
								{
									const real_t phiMy=Qdata(i, j-1, IH);
									if (fmin(phiMy, phi)>ZERO_F||fmax(phiMy, phi)<=ZERO_F)
									{
										HydroState qMy=getHydroState(Qdata, i, j-1);

										real_t rho0=qMy[ID], p0=qMy[IP];
										real_t delta_j=-1;
										const real_t vn    =qMy[IU]*gradphix    +qMy[IV]*gradphiy;
										const real_t vt    =qMy[IV]*gradphix    -qMy[IU]*gradphiy;
										const real_t num=ONE_F;
										const real_t dp=g_y*dy*delta_j;
										if (!barotropic)
										{
											p0-=dp*rho0;
										}
										else
										{
											const real_t pMy = computePressure(qMy);
											qMy[IP] =pMy - dp * rho0;
											rho0 = computeDensity(qMy);
										}

										rho+=rho0*num;
										u+=vn*num;
										v+=vt*num;
										p+=p0*num;
										number+=num;
									}
								}
								if (j<params.jmax-ghostWidth)
								{
									const real_t phiPy=Qdata(i, j+1, IH);
									if (fmin(phiPy, phi)>ZERO_F||fmax(phiPy, phi)<=ZERO_F)
									{
										HydroState qPy=getHydroState(Qdata, i, j+1);

										real_t rho0=qPy[ID], p0=qPy[IP];
										real_t delta_j=1;
										const real_t vn    =qPy[IU]*gradphix    +qPy[IV]*gradphiy;
										const real_t vt    =qPy[IV]*gradphix    -qPy[IU]*gradphiy;
										const real_t num=ONE_F;
										const real_t dp=g_y*dy*delta_j;
										if (!barotropic)
										{
											p0-=dp*rho0;
										}
										else
										{
											const real_t pPy = computePressure(qPy);
											qPy[IP] =pPy - dp * rho0;
											rho0 = computeDensity(qPy);
										}

										rho+=rho0*num;
										u+=vn*num;
										v+=vt*num;
										p+=p0*num;
										number+=num;
									}
								}
								if (number>0)
								{
									rho/=number;
									p/=number;
									u/=number;
									v/=number;


									const real_t vnLoc    =qLoc[IU]*gradphix    +qLoc[IV]*gradphiy;
									const real_t vtLoc    =qLoc[IV]*gradphix    -qLoc[IU]*gradphiy;
									if (params.settings.sigma==ZERO_F)
                                                                        {
                                                                             u = vnLoc;
									     v = params.settings.mu0>ZERO_F? vtLoc : v;
                                           
                                                                        }
									const real_t ux=(gradphix *      u   -   v    * gradphiy);
									const real_t uy=(gradphix *      v    +  u    * gradphiy);

									if ((!barotropic && barotropic0) ||(!barotropic0 && barotropic))
									{
										if (!barotropic0)
										{
											qLoc[IP]=computeTemperature(qLoc);
										}
										else
										{
											if (params.settings.sigma>ZERO_F)
											{
												qLoc[IP] = p; 
											}
											else
											{
												qLoc[IP]=computePressure(qLoc);
											}
										}
									}
									else
									{
										if (!barotropic0)
										{
											if (params.settings.sigma>ZERO_F)
											{
												qLoc[IP] = p; 
											}
										}

									}
									qLoc[IU]=ux;
									qLoc[IV]=uy;
									qLoc[ID]=rho;
									qLoc[IH]=phi;

									setHydroState(Udata, computeConservatives(qLoc), i, j);


								}

							}

						}
					}
				const DataArray Udata;
				const DataArray Qdata;
				const DataArray gradphi;
		};//change state


		class ComputeViscosityStepFunctor2D : HydroBaseFunctor2D
		{
			public:
				ComputeViscosityStepFunctor2D(HydroParams params_, DataArray Udata_, DataArrayConst Qdata_, real_t dt_):
					HydroBaseFunctor2D(params_), Udata(Udata_), Qdata(Qdata_), dt(dt_) {};

				static void apply(HydroParams params,
						DataArray Udata, DataArrayConst Qdata,
						real_t dt, int nbCells)
				{
					ComputeViscosityStepFunctor2D functor(params, Udata, Qdata, dt);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{

						const int ghostWidth = params.ghostWidth;
						int i, j;
						index2coord(index, i, j, params.isize, params.jsize);

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{


							real_t phi = Qdata(i, j, IH);
							const real_t muLoc=phi>ZERO_F? params.settings.mu0: params.settings.mu1;
							const real_t mu = fabs(phi)<params.dx? HALF_F*(params.settings.mu0+ params.settings.mu1) : muLoc;

							const real_t lambda = ZERO_F;
							const real_t eta = lambda - TWO_F/(ONE_F+TWO_F) * mu;

							const bool   barotropic= phi>ZERO_F? params.settings.barotropic0 : params.settings.barotropic1;
							const real_t FOUR_F = TWO_F * TWO_F;

							const real_t dx = params.dx;
							const real_t dy = params.dy;
							const real_t dtdx = dt / dx;
							const real_t dtdy = dt / dy;

							{

								const HydroState qMx   = getHydroState(Qdata, i-1, j  );
								const HydroState qPx   = getHydroState(Qdata, i  , j  );
								const HydroState qMxMy = getHydroState(Qdata, i-1, j-1);
								const HydroState qMxPy = getHydroState(Qdata, i-1, j+1);
								const HydroState qPxPy = getHydroState(Qdata, i  , j+1);
								const HydroState qPxMy = getHydroState(Qdata, i  , j-1);

								const real_t uInterface = (qMx[IU] + qPx[IU]) / TWO_F;
								const real_t vInterface = (qMx[IV] + qPx[IV]) / TWO_F;
								const real_t uCornerUp   = (qMxPy[IU] + qPxPy[IU] + qMx[IU] + qPx[IU]) / FOUR_F;
								const real_t uCornerDown = (qMx[IU] + qPx[IU] + qMxMy[IU] + qPxMy[IU]) / FOUR_F;
								const real_t vCornerUp   = (qMxPy[IV] + qPxPy[IV] + qMx[IV] + qPx[IV]) / FOUR_F;
								const real_t vCornerDown = (qMx[IV] + qPx[IV] + qMxMy[IV] + qPxMy[IV]) / FOUR_F;

								const real_t du_dx = (qPx[IU] - qMx[IU]) / dx;
								const real_t dv_dx = (qPx[IV] - qMx[IV]) / dx;
								const real_t du_dy = (uCornerUp - uCornerDown) / dy;
								const real_t dv_dy = (vCornerUp - vCornerDown) / dy;

								// Compute fluxes
								const real_t tau_xx = TWO_F * mu * du_dx + eta * (du_dx + dv_dy);
								const real_t tau_xy = mu * (du_dy + dv_dx);

								// Update the right cell of the interface
								Udata(i, j, IU) += - dtdx * tau_xx;
								Udata(i, j, IV) += - dtdx * tau_xy;

								if (!barotropic)
									Udata(i, j, IP) += - dtdx * (uInterface * tau_xx + vInterface * tau_xy);
							}

							{
								const HydroState qMx   = getHydroState(Qdata, i  , j  );
								const HydroState qPx   = getHydroState(Qdata, i+1, j  );
								const HydroState qMxMy = getHydroState(Qdata, i  , j-1);
								const HydroState qMxPy = getHydroState(Qdata, i  , j+1);
								const HydroState qPxPy = getHydroState(Qdata, i+1, j+1);
								const HydroState qPxMy = getHydroState(Qdata, i+1, j-1);

								const real_t uInterface = (qMx[IU] + qPx[IU]) / TWO_F;
								const real_t vInterface = (qMx[IV] + qPx[IV]) / TWO_F;
								const real_t uCornerUp   = (qMxPy[IU] + qPxPy[IU] + qMx[IU] + qPx[IU]) / FOUR_F;
								const real_t uCornerDown = (qMx[IU] + qPx[IU] + qMxMy[IU] + qPxMy[IU]) / FOUR_F;
								const real_t vCornerUp   = (qMxPy[IV] + qPxPy[IV] + qMx[IV] + qPx[IV]) / FOUR_F;
								const real_t vCornerDown = (qMx[IV] + qPx[IV] + qMxMy[IV] + qPxMy[IV]) / FOUR_F;

								const real_t du_dx = (qPx[IU] - qMx[IU]) / dx;
								const real_t dv_dx = (qPx[IV] - qMx[IV]) / dx;
								const real_t du_dy = (uCornerUp - uCornerDown) / dy;
								const real_t dv_dy = (vCornerUp - vCornerDown) / dy;

								// Compute fluxes
								const real_t tau_xx = TWO_F * mu * du_dx + eta * (du_dx + dv_dy);
								const real_t tau_xy = mu * (du_dy + dv_dx);

								// Update the right cell of the interface
								Udata(i, j, IU) +=   dtdx * tau_xx;
								Udata(i, j, IV) +=   dtdx * tau_xy;

								if (!barotropic)
									Udata(i, j, IP) +=   dtdx * (uInterface * tau_xx + vInterface * tau_xy);
							}

							{
								const HydroState qMy   = getHydroState(Qdata, i  , j-1);
								const HydroState qPy   = getHydroState(Qdata, i  , j  );
								const HydroState qMxMy = getHydroState(Qdata, i-1, j-1);
								const HydroState qMxPy = getHydroState(Qdata, i-1, j  );
								const HydroState qPxPy = getHydroState(Qdata, i+1, j  );
								const HydroState qPxMy = getHydroState(Qdata, i+1, j-1);

								const real_t uInterface = (qMy[IU] + qPy[IU]) / TWO_F;
								const real_t vInterface = (qMy[IV] + qPy[IV]) / TWO_F;
								const real_t uCornerLeft  = (qMxPy[IU] + qPy[IU] + qMxMy[IU] + qMy[IU]) / FOUR_F;
								const real_t uCornerRight = (qPy[IU] + qPxPy[IU] + qMy[IU] + qPxMy[IU]) / FOUR_F;
								const real_t vCornerLeft  = (qMxPy[IV] + qPy[IV] + qMxMy[IV] + qMy[IV]) / FOUR_F;
								const real_t vCornerRight = (qPy[IV] + qPxPy[IV] + qMy[IV] + qPxMy[IV]) / FOUR_F;

								const real_t du_dx = (uCornerRight - uCornerLeft) / dx;
								const real_t dv_dx = (vCornerRight - vCornerLeft) / dx;
								const real_t du_dy = (qPy[IU] - qMy[IU]) / dy;
								const real_t dv_dy = (qPy[IV] - qMy[IV]) / dy;

								const real_t tau_yy = TWO_F * mu * dv_dy + eta * (du_dx + dv_dy);
								const real_t tau_xy = mu * (du_dy + dv_dx);

								// Update the up cell of the interface
								Udata(i, j, IU) += - dtdy * tau_xy;
								Udata(i, j, IV) += - dtdy * tau_yy;
								if (!barotropic)
									Udata(i, j, IP) += - dtdy * (uInterface * tau_xy + vInterface * tau_yy);
							}

							{
								const HydroState qMy   = getHydroState(Qdata, i  , j  );
								const HydroState qPy   = getHydroState(Qdata, i  , j+1);
								const HydroState qMxMy = getHydroState(Qdata, i-1, j  );
								const HydroState qMxPy = getHydroState(Qdata, i-1, j+1);
								const HydroState qPxPy = getHydroState(Qdata, i+1, j+1);
								const HydroState qPxMy = getHydroState(Qdata, i+1, j  );

								const real_t uInterface = (qMy[IU] + qPy[IU]) / TWO_F;
								const real_t vInterface = (qMy[IV] + qPy[IV]) / TWO_F;
								const real_t uCornerLeft  = (qMxPy[IU] + qPy[IU] + qMxMy[IU] + qMy[IU]) / FOUR_F;
								const real_t uCornerRight = (qPy[IU] + qPxPy[IU] + qMy[IU] + qPxMy[IU]) / FOUR_F;
								const real_t vCornerLeft  = (qMxPy[IV] + qPy[IV] + qMxMy[IV] + qMy[IV]) / FOUR_F;
								const real_t vCornerRight = (qPy[IV] + qPxPy[IV] + qMy[IV] + qPxMy[IV]) / FOUR_F;

								const real_t du_dx = (uCornerRight - uCornerLeft) / dx;
								const real_t dv_dx = (vCornerRight - vCornerLeft) / dx;
								const real_t du_dy = (qPy[IU] - qMy[IU]) / dy;
								const real_t dv_dy = (qPy[IV] - qMy[IV]) / dy;

								const real_t tau_yy = TWO_F * mu * dv_dy + eta * (du_dx + dv_dy);
								const real_t tau_xy = mu * (du_dy + dv_dx);

								// Update the bottom cell of the interface
								Udata(i, j, IU) +=   dtdy * tau_xy;
								Udata(i, j, IV) +=   dtdy * tau_yy;
								if (!barotropic)
									Udata(i, j, IP) +=   dtdy * (uInterface * tau_xy + vInterface * tau_yy);
							}

						}

					}

				const DataArray Udata;
				const DataArrayConst Qdata;
				const real_t dt;
		}; // ComputeViscosityStepFunctor2D


		class ComputeHeatDiffusionStepFunctor2D : HydroBaseFunctor2D
		{
			public:
				ComputeHeatDiffusionStepFunctor2D(HydroParams params_, DataArray Udata_, DataArrayConst Qdata_, real_t dt_):
					HydroBaseFunctor2D(params_), Udata(Udata_), Qdata(Qdata_), dt(dt_) {};

				static void apply(HydroParams params,
						DataArray Udata, DataArrayConst Qdata,
						real_t dt, int nbCells)
				{
					ComputeHeatDiffusionStepFunctor2D functor(params, Udata, Qdata, dt);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						const int ghostWidth = params.ghostWidth;
						const real_t dx = params.dx;
						const real_t dy = params.dy;
						const real_t dtdx = dt / dx;
						const real_t dtdy = dt / dy;

						int i, j;
						index2coord(index, i, j, params.isize, params.jsize);

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const HydroState qLoc = getHydroState(Qdata, i, j);
							const real_t TLoc = computeTemperature(qLoc);
							const real_t kappa=qLoc[IH]>ZERO_F? params.settings.kappa0:params.settings.kappa1;
							const bool   barotropic= qLoc[IH]>ZERO_F? params.settings.barotropic0 : params.settings.barotropic1;

							real_t energy_fluxes = ZERO_F;
							{
								const HydroState qNei = getHydroState(Qdata, i-1, j);
								const real_t TNei = computeTemperature(qNei);
								if (qLoc[IH]* qNei[IH] <=ZERO_F && (qLoc[IH] >ZERO_F || qNei[IH] > ZERO_F ))
								{
									const real_t kappa_Nei=qNei[IH]>ZERO_F? params.settings.kappa0:params.settings.kappa1;
									const real_t dxLoc = dx * fabs(qLoc[IH]) / fabs(qLoc[IH] -qNei[IH]);
									const real_t dxNei = dx * fabs(qNei[IH]) / fabs(qLoc[IH] -qNei[IH]);
									real_t T0 = (kappa * TLoc * dxNei + kappa_Nei * TNei * dxLoc) / (kappa * dxNei + kappa_Nei * dxLoc);
									energy_fluxes += dtdx * kappa * (T0 - TLoc) / dxLoc;

								}
								else
									energy_fluxes += dtdx * kappa * (TNei - TLoc) / dx;
							}

							{
								const HydroState qNei = getHydroState(Qdata, i+1, j);
								const real_t TNei = computeTemperature(qNei);
								if (qLoc[IH]* qNei[IH] <=ZERO_F && (qLoc[IH] >ZERO_F || qNei[IH] > ZERO_F ))
								{
									const real_t kappa_Nei=qNei[IH]>ZERO_F? params.settings.kappa0:params.settings.kappa1;
									const real_t dxLoc = dx * fabs(qLoc[IH]) / fabs(qLoc[IH] -qNei[IH]);
									const real_t dxNei = dx * fabs(qNei[IH]) / fabs(qLoc[IH] -qNei[IH]);
									real_t T0 = (kappa * TLoc * dxNei + kappa_Nei * TNei * dxLoc) / (kappa * dxNei + kappa_Nei * dxLoc);
									energy_fluxes += dtdx * kappa * (T0 - TLoc) / dxLoc;
								}
								else
									energy_fluxes += dtdx * kappa * (TNei - TLoc) / dx;
							}

							{
								const HydroState qNei = getHydroState(Qdata, i, j-1);
								const real_t TNei = computeTemperature(qNei);
								if (qLoc[IH]* qNei[IH] <=ZERO_F && (qLoc[IH] >ZERO_F || qNei[IH] > ZERO_F ))
								{
									const real_t kappa_Nei=qNei[IH]>ZERO_F? params.settings.kappa0:params.settings.kappa1;
									const real_t dyLoc = dy * fabs(qLoc[IH]) / fabs(qLoc[IH] -qNei[IH]);
									const real_t dyNei = dy * fabs(qNei[IH]) / fabs(qLoc[IH] -qNei[IH]);
									real_t T0 = (kappa * TLoc * dyNei + kappa_Nei * TNei * dyLoc) / (kappa * dyNei + kappa_Nei * dyLoc);
									energy_fluxes += dtdy * kappa * (T0 - TLoc) / dyLoc;
								}
								else
								{
									energy_fluxes += dtdy * kappa * (TNei - TLoc) / dy;
								}
							}

							{
								const HydroState qNei = getHydroState(Qdata, i, j+1);
								const real_t TNei = computeTemperature(qNei);
								if (qLoc[IH]* qNei[IH] <=ZERO_F && (qLoc[IH] >ZERO_F || qNei[IH] > ZERO_F ))
								{
									const real_t kappa_Nei=qNei[IH]>ZERO_F? params.settings.kappa0:params.settings.kappa1;
									const real_t dyLoc = dy * fabs(qLoc[IH]) / fabs(qLoc[IH] -qNei[IH]);
									const real_t dyNei = dy * fabs(qNei[IH]) / fabs(qLoc[IH] -qNei[IH]);
									real_t T0 = (kappa * TLoc * dyNei + kappa_Nei * TNei * dyLoc) / (kappa * dyNei + kappa_Nei * dyLoc);
									energy_fluxes += dtdy * kappa * (T0 - TLoc) / dyLoc;

								}
								else
								{
									energy_fluxes += dtdy * kappa * (TNei - TLoc) / dy;
								}
							}

							if (!barotropic)
								Udata(i, j, IE) += energy_fluxes;
							else
							{
								const real_t Rstar = qLoc[IH]>ZERO_F? params.settings.Rstar0 : params.settings.Rstar1;
								Udata(i, j, IE) += energy_fluxes / qLoc[ID] / Rstar;

							}

						}
					}

				const DataArray Udata;
				const DataArrayConst Qdata;
				const real_t dt;
		}; // ComputeHeatDiffusionStepFunctor2D


		class ComputeDtFunctor2D : public HydroBaseFunctor2D
		{
			public:
				ComputeDtFunctor2D(HydroParams params_, DataArrayConst Udata_) :
					HydroBaseFunctor2D(params_), Udata(Udata_)  {};

				static void apply(HydroParams params, DataArrayConst Udata, real_t& invDt, int nbCells)
				{
					ComputeDtFunctor2D functor(params, Udata);
					Kokkos::parallel_reduce(nbCells, functor, invDt);
				}

				// Tell each thread how to initialize its reduction result.
				KOKKOS_INLINE_FUNCTION
					void init (real_t& dst) const
					{
						// The identity under max is -Inf.
						// Kokkos does not come with a portable way to access
						// floating-point Inf and NaN.
#ifdef __CUDA_ARCH__
						dst = -CUDART_INF;
#else
						dst = std::numeric_limits<real_t>::min();
#endif // __CUDA_ARCH__
					} // init

				/* this is a reduce (max) functor */
				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index, real_t& invDt) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ghostWidth = params.ghostWidth;

						const real_t dx = params.dx;
						const real_t dy = params.dy;

						int i,j;
						index2coord(index,i,j,isize,jsize);

						if(j >= ghostWidth && j < jsize - ghostWidth &&
								i >= ghostWidth && i < isize - ghostWidth)
						{
							// get local conservative variable
							const HydroState uLoc = getHydroState(Udata, i, j);
							// get primitive variables in current cell
							const HydroState qLoc = computePrimitives(uLoc);
							const real_t c = computeSpeedSound(qLoc);
							const real_t vx = c+FABS(qLoc[IU]);
							const real_t vy = c+FABS(qLoc[IV]);

							// Hyperbolic part
							invDt = FMAX(invDt, vx/dx + vy/dy);
							// Viscous flux
							// (1.0*mu+abs(-2.0/3.0*mu)=8.0/3.0*mu
							//  formula still needs some justification)
							invDt = FMAX(invDt, 8.0 / 3.0 * FMAX(params.settings.mu0, params.settings.mu1) / uLoc[ID] * (ONE_F/(dx*dx) + ONE_F/(dy*dy)));
							// Heat flux
							real_t kappa=qLoc[IH]>ZERO_F? params.settings.kappa0:params.settings.kappa1;
							real_t cp=qLoc[IH]>ZERO_F? params.settings.cp0:params.settings.cp1;
							invDt = FMAX(invDt, kappa / (uLoc[ID] * cp) * (ONE_F/(dx*dx) + ONE_F/(dy*dy)));
							//Surface tension
							const real_t pi=acos(-1.);
							invDt = FMAX(invDt, sqrt(4*pi*params.settings.sigma/(2. * uLoc[ID]*dx*dx*dx)));
						}
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

				const DataArrayConst Udata;
		}; // ComputeDtFunctor2D


		class ComputeAcousticDtFunctor2D : public HydroBaseFunctor2D
		{
			public:
				ComputeAcousticDtFunctor2D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor2D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& invDt, int nbCells)
				{
					ComputeAcousticDtFunctor2D functor(params, Qdata);
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
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ghostWidth = params.ghostWidth;

						const real_t dx = params.dx;
						const real_t dy = params.dy;

						int i, j;
						index2coord(index, i, j, isize, jsize);

						if(j >= ghostWidth && j <= jsize - ghostWidth &&
								i >= ghostWidth && i <= isize - ghostWidth)
						{
							const HydroState qLoc = getHydroState(Qdata, i, j);
							const real_t cLoc = computeSpeedSound(qLoc);
							const real_t K    = params.settings.K;

							if (j != jsize-ghostWidth)
							{
								const HydroState qNei = getHydroState(Qdata, i-1, j);
								const real_t cNei = computeSpeedSound(qNei);

								const real_t aNei = K * FMAX(qNei[ID] * cNei, qLoc[ID] * cLoc);
								const real_t invDtNei = aNei / FMIN(dx * qNei[ID], dx * qLoc[ID]);
								invDt = FMAX(invDt, invDtNei);
							}

							if (i != isize-ghostWidth)
							{
								const HydroState qNei = getHydroState(Qdata, i, j-1);
								const real_t cNei = computeSpeedSound(qNei);

								const real_t aNei = K * FMAX(qNei[ID] * cNei, qLoc[ID] * cLoc);
								const real_t invDtNei = aNei / FMIN(dy * qNei[ID], dy * qLoc[ID]);
								invDt = FMAX(invDt, invDtNei);
							}
						}
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
		}; // ComputeAcousticDtFunctor2D


		class ComputeTransportDtFunctor2D : public HydroBaseFunctor2D
		{
			public:
				ComputeTransportDtFunctor2D(HydroParams params_, DataArrayConst Qdata_) :
					HydroBaseFunctor2D(params_), Qdata(Qdata_)  {};

				static void apply(HydroParams params, DataArrayConst Qdata, real_t& invDt, int nbCells)
				{
					ComputeTransportDtFunctor2D functor(params, Qdata);
					Kokkos::parallel_reduce(nbCells, functor, invDt);
				}

				KOKKOS_INLINE_FUNCTION
					void computeAcousticRelaxation(const HydroState& qLoc, real_t cLoc,
							const HydroState& qNei, real_t cNei,
							real_t M, int IX, int dir, real_t& uStar) const
					{
						const real_t a = params.settings.K * FMAX(qNei[ID] * cNei, qLoc[ID] * cLoc);
						uStar = dir * HALF_F * (qNei[IX] + qLoc[IX]) - HALF_F * (qNei[IP] - qLoc[IP] + M) / a;
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
						const int isize = params.isize;
						const int jsize = params.jsize;
						const int ghostWidth = params.ghostWidth;

						const real_t dx = params.dx;
						const real_t dy = params.dy;

						int i,j;
						index2coord(index,i,j,isize,jsize);

						if(j >= ghostWidth && j <= jsize - ghostWidth &&
								i >= ghostWidth && i <= isize - ghostWidth)
						{
							const HydroState qLoc = getHydroState(Qdata, i, j);
							const real_t cLoc = computeSpeedSound(qLoc);
							const real_t psiLoc = psi(i, j);

							real_t invDtLoc = ZERO_F;

							{
								const HydroState qMx = getHydroState(Qdata, i-1, j);
								const real_t cMinusX = computeSpeedSound(qMx);
								const real_t psiMx = psi(i-1, j);
								const real_t Mmx = computeM(qLoc, psiLoc, qMx, psiMx);
								real_t uStar;
								computeAcousticRelaxation(qLoc, cLoc, qMx, cMinusX, Mmx, IU, -1,
										uStar);
								invDtLoc += FABS(uStar) / dx;
							}

							{
								const HydroState qPx = getHydroState(Qdata, i+1, j);
								const real_t cPlusX = computeSpeedSound(qPx);
								const real_t psiPx = psi(i+1, j);
								const real_t Mpx = computeM(qLoc, psiLoc, qPx, psiPx);
								real_t uStar;
								computeAcousticRelaxation(qLoc, cLoc, qPx, cPlusX, Mpx, IU, +1,
										uStar);
								invDtLoc += FABS(uStar) / dx;
							}

							{
								const HydroState qMy = getHydroState(Qdata, i, j-1);
								const real_t cMinusY = computeSpeedSound(qMy);
								const real_t psiMy = psi(i, j-1);
								const real_t Mmy = computeM(qLoc, psiLoc, qMy, psiMy);
								real_t uStar;
								computeAcousticRelaxation(qLoc, cLoc, qMy, cMinusY, Mmy, IV, -1,
										uStar);
								invDtLoc += FABS(uStar) / dy;
							}

							{
								const HydroState qPy = getHydroState(Qdata, i, j+1);
								const real_t cPlusY = computeSpeedSound(qPy);
								const real_t psiPy = psi(i, j+1);
								const real_t Mpy = computeM(qLoc, psiLoc, qPy, psiPy);
								real_t uStar;
								computeAcousticRelaxation(qLoc, cLoc, qPy, cPlusY, Mpy, IV, +1,
										uStar);
								invDtLoc += FABS(uStar) / dy;
							}

							invDt = FMAX(invDt, invDtLoc);
						}
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
		}; // ComputeTransportDtFunctor2D


		class ConvertToPrimitivesFunctor2D : public HydroBaseFunctor2D
		{
			public:
				ConvertToPrimitivesFunctor2D(HydroParams params_, DataArrayConst Udata_, DataArray Qdata_) :
					HydroBaseFunctor2D(params_), Udata(Udata_), Qdata(Qdata_)  {};

				static void apply(HydroParams params,
						DataArrayConst Udata, DataArray Qdata,
						int nbCells)
				{
					ConvertToPrimitivesFunctor2D functor(params, Udata, Qdata);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(const int& index) const
					{
						const int isize = params.isize;
						const int jsize = params.jsize;

						int i,j;
						index2coord(index,i,j,isize,jsize);

						if(j >= 0 && j < jsize  && i >= 0 && i < isize)
						{
							// get local conservative variable
							const HydroState uLoc = getHydroState(Udata, i, j);
							// get primitive variables in current cell
							const HydroState qLoc = computePrimitives(uLoc);
							// copy q state in q global
							setHydroState(Qdata, qLoc, i, j);
						}
					}

				const DataArrayConst Udata;
				const DataArray Qdata;
		}; // ConvertToPrimitivesFunctor2D
		class CopyGradientFunctor2D : HydroBaseFunctor2D
		{
			public:
				CopyGradientFunctor2D(HydroParams params_,
						DataArray Qdata_,const int KX_, DataArray Udata_, const int KY_) :
					HydroBaseFunctor2D(params_),
					Qdata(Qdata_), KX(KX_), Udata(Udata_), KY(KY_) {};

				static void apply(HydroParams params,
						DataArray Qdata, const int KX, DataArray Udata, const int KY,
						int nbCells)
				{
					CopyGradientFunctor2D functor(params, Qdata, KX, Udata, KY);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);

						const int ghostWidth = params.ghostWidth;

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							Udata(i, j, KY) = Qdata(i, j, KX);

						}
					}

				const DataArray Qdata;
				const int KX;
				const DataArray Udata;
				const int KY;
				const DataArray gradphi;
		}; // ComputeGradientUVFunctor2D
		class ComputeExtraUVFunctor2D : HydroBaseFunctor2D
		{
			public:
				ComputeExtraUVFunctor2D(HydroParams params_,
						DataArrayConst Qdata_, DataArray Q0data_, DataArray Udata_, DataArray gradphi_) :
					HydroBaseFunctor2D(params_),
					Qdata(Qdata_), Q0data(Q0data_), Udata(Udata_), gradphi(gradphi_) {};

				static void apply(HydroParams params,
						DataArrayConst Qdata, DataArray Q0data, DataArray Udata, DataArray gradphi,
						int nbCells)
				{
					ComputeExtraUVFunctor2D functor(params, Qdata, Q0data, Udata, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);

						const int ghostWidth = params.ghostWidth;
						const int imin = ghostWidth;
						const int jmin = ghostWidth;
						const int imax = params.imax-ghostWidth;
						const int jmax = params.jmax-ghostWidth;
						const real_t dx = params.dx;
						const real_t dy = params.dy;

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t phiLoc = Qdata(i, j, IH);
							if (fabs(phiLoc)<=dx)
							{
								int delta_iM = phiLoc * gradphi(i, j, IPX) < ZERO_F || (phiLoc==ZERO_F && gradphi(i, j, IPX)>ZERO_F)? +1 : -1; 
								const real_t phi_deltaiM = Qdata(i+delta_iM, j, IH);
								if (i+ delta_iM < imin || i + delta_iM > imax || fmax(phiLoc, phi_deltaiM)<=ZERO_F  \
										|| fmin(phiLoc, phi_deltaiM)>ZERO_F || fabs(gradphi(i, j, IPX))<1.e-10)
									delta_iM = 0;

								int delta_jM = phiLoc * gradphi(i, j, IPY) < ZERO_F || (phiLoc==ZERO_F && gradphi(i, j, IPY)>ZERO_F)? +1 : -1; 
								const real_t phi_deltajM = Qdata(i, j+delta_jM, IH);
								if (j+ delta_jM < jmin || j + delta_jM > jmax || fmax(phiLoc, phi_deltajM)<=ZERO_F  \
										|| fmin(phiLoc, phi_deltajM)>ZERO_F || fabs(gradphi(i, j, IPY))<1.e-10)
									delta_jM = 0;

								if (delta_iM!=0 || delta_jM!=0) 
								{
									const real_t gradux = delta_iM != 0 ?(Qdata(i+delta_iM, j, IU) - Q0data(i, j, IU))/(dx * delta_iM) : ZERO_F;
									const real_t graduy = delta_jM != 0 ?(Qdata(i, j+delta_jM, IU) - Q0data(i, j, IU))/(dy * delta_jM) : ZERO_F;
									const real_t sign = phiLoc > ZERO_F ? ONE_F : -ONE_F;
									Udata(i, j, ID) = Q0data(i, j, IU) - 0.8 * dx * sign * (gradphi(i, j, IPX) * gradux + gradphi(i, j, IPY) * graduy - Udata(i, j, IU));

									const real_t gradvx = delta_iM != 0 ?(Qdata(i+delta_iM, j, IV) - Q0data(i, j, IV))/(dx * delta_iM) : ZERO_F;
									const real_t gradvy = delta_jM != 0 ?(Qdata(i, j+delta_jM, IV) - Q0data(i, j, IV))/(dy * delta_jM) : ZERO_F;

									Udata(i, j, IE) = Q0data(i, j, IV) - 0.8 * dx * sign * (gradphi(i, j, IPX) * gradvx + gradphi(i, j, IPY) * gradvy - Udata(i, j, IV));


								}


							}

						}
					}

				const DataArrayConst Qdata;
				const DataArray Q0data;
				const DataArray Udata;
				const DataArray gradphi;
		}; // ComputeGradientFunctor2D

		class ComputeExtraGradientUVFunctor2D : HydroBaseFunctor2D
		{
			public:
				ComputeExtraGradientUVFunctor2D(HydroParams params_,
						DataArrayConst Qdata_, DataArray Udata_, DataArray gradphi_) :
					HydroBaseFunctor2D(params_),
					Qdata(Qdata_), Udata(Udata_), gradphi(gradphi_) {};

				static void apply(HydroParams params,
						DataArrayConst Qdata,  DataArray Udata, DataArray gradphi,
						int nbCells)
				{
					ComputeExtraGradientUVFunctor2D functor(params, Qdata,  Udata, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);

						const int ghostWidth = params.ghostWidth;
						const int imin = ghostWidth;
						const int jmin = ghostWidth;
						const int imax = params.imax-ghostWidth;
						const int jmax = params.jmax-ghostWidth;
						const real_t dx = params.dx;
						const real_t dy = params.dy;

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t phiLoc = Qdata(i, j, IH);
							if (fabs(phiLoc)<=dx)
							{
								int delta_iM = phiLoc * gradphi(i, j, IPX) < ZERO_F || (phiLoc==ZERO_F && gradphi(i, j, IPX)>ZERO_F)? +1 : -1; 
								const real_t phi_deltaiM = Qdata(i+delta_iM, j, IH);
								if (i+ delta_iM < imin || i + delta_iM > imax || fmax(phiLoc, phi_deltaiM)<=ZERO_F  \
										|| fmin(phiLoc, phi_deltaiM)>ZERO_F || fabs(gradphi(i, j, IPX))<1.e-10)
									delta_iM = 0;

								int delta_jM = phiLoc * gradphi(i, j, IPY) < ZERO_F || (phiLoc==ZERO_F && gradphi(i, j, IPY)>ZERO_F)? +1 : -1; 
								const real_t phi_deltajM = Qdata(i, j+delta_jM, IH);
								if (j+ delta_jM < jmin || j + delta_jM > jmax || fmax(phiLoc, phi_deltajM)<=ZERO_F  \
										|| fmin(phiLoc, phi_deltajM)>ZERO_F || fabs(gradphi(i, j, IPY))<1.e-10)
									delta_jM = 0;

								if (delta_iM!=0 || delta_jM!=0) 
								{
									const real_t gradux = delta_iM != 0 ?(gradphi(i+delta_iM, j, IPS) - Udata(i, j, IU))/(dx * delta_iM) : ZERO_F;
									const real_t graduy = delta_jM != 0 ?(gradphi(i, j+delta_jM, IPS) - Udata(i, j, IU))/(dy * delta_jM) : ZERO_F;
									const real_t sign = phiLoc > ZERO_F ? ONE_F : -ONE_F;
									Udata(i, j, ID) = Udata(i, j, IU) - 0.8 * dx * sign * (gradphi(i, j, IPX) * gradux + gradphi(i, j, IPY) * graduy );

									const real_t gradvx = delta_iM != 0 ?(gradphi(i+delta_iM, j, IPK) - Udata(i, j, IV))/(dx * delta_iM) : ZERO_F;
									const real_t gradvy = delta_jM != 0 ?(gradphi(i, j+delta_jM, IPK) - Udata(i, j, IV))/(dy * delta_jM) : ZERO_F;
									Udata(i, j, IE) = Udata(i, j, IV) - 0.8 * dx * sign * (gradphi(i, j, IPX) * gradvx + gradphi(i, j, IPY) * gradvy );

								}


							}

						}
					}

				const DataArrayConst Qdata;
				const DataArray Udata;
				const DataArray gradphi;
		}; // ComputeGradientUVFunctor2D

		class ComputeGradientUVFunctor2D : HydroBaseFunctor2D
		{
			public:
				ComputeGradientUVFunctor2D(HydroParams params_,
						DataArrayConst Qdata_, DataArray Q0data_, DataArray Udata_, DataArray gradphi_) :
					HydroBaseFunctor2D(params_),
					Qdata(Qdata_), Q0data(Q0data_), Udata(Udata_), gradphi(gradphi_) {};

				static void apply(HydroParams params,
						DataArrayConst Qdata, DataArray Q0data, DataArray Udata, DataArray gradphi,
						int nbCells)
				{
					ComputeGradientUVFunctor2D functor(params, Qdata, Q0data, Udata, gradphi);
					Kokkos::parallel_for(nbCells, functor);
				}

				KOKKOS_INLINE_FUNCTION
					void operator()(int index) const
					{
						int i,j;
						index2coord(index, i, j, params.isize, params.jsize);

						const int ghostWidth = params.ghostWidth;
						const int imin = ghostWidth;
						const int jmin = ghostWidth;
						const int imax = params.imax-ghostWidth;
						const int jmax = params.jmax-ghostWidth;
						const real_t dx = params.dx;
						const real_t dy = params.dy;

						if (j>=ghostWidth && j<=params.jmax-ghostWidth &&
								i>=ghostWidth && i<=params.imax-ghostWidth)
						{
							const real_t phiLoc = Qdata(i, j, IH);
							if (fabs(phiLoc)<=dx)
							{
								int delta_iM = phiLoc * gradphi(i, j, IPX) < ZERO_F || (phiLoc==ZERO_F && gradphi(i, j, IPX)>ZERO_F)? +1 : -1; 
								const real_t phi_deltaiM = Qdata(i+delta_iM, j, IH);
								if (i+ delta_iM < imin || i + delta_iM > imax || fmax(phiLoc, phi_deltaiM)<=ZERO_F  \
										|| fmin(phiLoc, phi_deltaiM)>ZERO_F || fabs(gradphi(i, j, IPX))<1.e-10)
									delta_iM = 0;

								int delta_jM = phiLoc * gradphi(i, j, IPY) < ZERO_F || (phiLoc==ZERO_F && gradphi(i, j, IPY)>ZERO_F)? +1 : -1; 
								const real_t phi_deltajM = Qdata(i, j+delta_jM, IH);
								if (j+ delta_jM < jmin || j + delta_jM > jmax || fmax(phiLoc, phi_deltajM)<=ZERO_F  \
										|| fmin(phiLoc, phi_deltajM)>ZERO_F || fabs(gradphi(i, j, IPY))<1.e-10)
									delta_jM = 0;


								if (delta_iM!=0 || delta_jM!=0) 
								{
									int delta_i = phiLoc * gradphi(i, j, IPX) > ZERO_F || (phiLoc==ZERO_F && gradphi(i, j, IPX)<ZERO_F)? +1 : -1; 
									const real_t phi_deltai = Qdata(i+delta_i, j, IH);
									if (i+ delta_i < imin || i + delta_i > imax || (fmax(phiLoc, phi_deltai)>ZERO_F  \
												&& fmin(phiLoc, phi_deltai)<=ZERO_F)||fabs(gradphi(i, j, IPX))<1.e-10)
										delta_i = 0;

									int delta_j = phiLoc * gradphi(i, j, IPY) > ZERO_F || (phiLoc==ZERO_F && gradphi(i, j, IPY)<ZERO_F)? +1 : -1; 
									const real_t phi_deltaj = Qdata(i, j+delta_j, IH);
									if (j+ delta_j < jmin || j + delta_j > jmax || (fmax(phiLoc, phi_deltaj)>ZERO_F \
												&& fmin(phiLoc, phi_deltaj)<=ZERO_F)||fabs(gradphi(i, j, IPY))<1.e-10)
										delta_j = 0;

									if (delta_i==0 && delta_j==0)
										std::cout<<" Error"<<std::endl; 

									const real_t gradux = delta_i != 0 ?(Qdata(i+delta_i, j, IU) - Qdata(i, j, IU))/(dx * delta_i) : ZERO_F;
									const real_t gradvx = delta_i != 0 ?(Qdata(i+delta_i, j, IV) - Qdata(i, j, IV))/(dx * delta_i) : ZERO_F;
									const real_t graduy = delta_j != 0 ?(Qdata(i, j+delta_j, IU) - Qdata(i, j, IU))/(dy * delta_j) : ZERO_F;
									const real_t gradvy = delta_j != 0 ?(Qdata(i, j+delta_j, IV) - Qdata(i, j, IV))/(dy * delta_j) : ZERO_F;

									gradphi(i, j, IPS) = (gradux * gradphi(i, j, IPX) + graduy * gradphi(i, j, IPY)); 

									gradphi(i, j, IPK) = (gradvx * gradphi(i, j, IPX) + gradvy * gradphi(i, j, IPY)); 

									if (delta_iM != 0)
									{ 
										Udata(i+delta_iM, j, ID) = gradphi(i, j, IPS);
										Udata(i+delta_iM, j, IU) = gradphi(i, j, IPS);

										Udata(i+delta_iM, j, IE) = gradphi(i, j, IPK);
										Udata(i+delta_iM, j, IV) = gradphi(i, j, IPK);

										Q0data(i+delta_iM, j, IU)= Qdata(i, j, IU);
										Q0data(i+delta_iM, j, IV)= Qdata(i, j, IV);
									}
									else if (delta_jM != 0)
									{
										Udata(i, j+delta_jM, ID) = gradphi(i, j, IPS);
										Udata(i, j+delta_jM, IU) = gradphi(i, j, IPS);

										Udata(i, j+delta_jM, IE) = gradphi(i, j, IPK);
										Udata(i, j+delta_jM, IV) = gradphi(i, j, IPK);

										Q0data(i, j+delta_jM, IU)= Qdata(i, j, IU);
										Q0data(i, j+delta_jM, IV)= Qdata(i, j, IV);
									}

									Udata(i, j, IH)=ONE_F;
								}
								else
									Udata(i, j, IH)=-ONE_F;


							}

						}
					}

				const DataArrayConst Qdata;
				const DataArray Q0data;
				const DataArray Udata;
				const DataArray gradphi;
		}; // ComputeGradientUVFunctor2D

	} // namespace all_regime

} // namespace euler_kokkos
