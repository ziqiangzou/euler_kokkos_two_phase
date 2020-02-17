#pragma once

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"
#include "shared/units.h"
#include <iomanip>


namespace euler_kokkos { namespace all_regime
	{

		/**
		 * Base class to derive actual kokkos functor for hydro 2D.
		 * params is passed by copy.
		 */
		class HydroBaseFunctor2D
		{
			public:
				using HydroState     = HydroState2d;
				using DataArray      = DataArray2d;
				using DataArrayConst = DataArray2dConst;

				HydroBaseFunctor2D(HydroParams params_) :
					params(params_), nbvar(params_.nbvar) {};
				virtual ~HydroBaseFunctor2D() {};

				const HydroParams params;
				const int nbvar;

				/**
				 * Compute gravitational potential at position (x, y)
				 * @param[in]  x    x-coordinate
				 * @param[in]  y    y-coordinate
				 * @param[out] psi  gravitational potential
				 */
				KOKKOS_INLINE_FUNCTION
					real_t psi(real_t x, real_t y) const
					{
						return - params.settings.g_x * x - params.settings.g_y * y;
					} // psi

				/**
				 * Compute gravitational potential at the center
				 * of the cell C(i, j)
				 * @param[in]  i    logical x-coordinate of the cell C
				 * @param[in]  j    logical y-coordinate of the cell C
				 * @param[out] psi  gravitational potential
				 */
				KOKKOS_INLINE_FUNCTION
					real_t psi(int i, int j) const
					{
#ifdef USE_MPI
						const int nx_mpi = params.nx * params.myMpiPos[IX];
						const int ny_mpi = params.ny * params.myMpiPos[IY];
						const real_t x = params.xmin + (HALF_F + i + nx_mpi - params.ghostWidth)*params.dx;
						const real_t y = params.ymin + (HALF_F + j + ny_mpi - params.ghostWidth)*params.dy;
#else
						const real_t x = params.xmin + (HALF_F + i - params.ghostWidth)*params.dx;
						const real_t y = params.ymin + (HALF_F + j - params.ghostWidth)*params.dy;
#endif

						return psi(x, y);
					} // psi

				/**
				 * Compute \mathcal{M} = \rho * \Delta \psi between two cells
				 * (to be renamed)
				 */
				KOKKOS_INLINE_FUNCTION
					real_t computeM(const HydroState& qLoc, real_t psiLoc,
							const HydroState& qNei, real_t psiNei) const
					{
						return HALF_F * (qLoc[ID]+qNei[ID]) * (psiNei - psiLoc);
					} // computeM

				/**
				 * Get HydroState from global array (either conservative or primitive)
				 * at cell C(i, j) (global to local)
				 * @param[in]  array global array
				 * @param[in]  i     logical x-coordinate of the cell C
				 * @param[in]  j     logical y-coordinate of the cell C
				 * @param[out] state HydroState of cell C(i, j)
				 */
				KOKKOS_INLINE_FUNCTION
					HydroState getHydroState(DataArrayConst array, int i, int j) const
					{
						HydroState state;
						state[ID] = array(i, j, ID);
						state[IP] = array(i, j, IP);
						state[IS] = array(i, j, IS);
						state[IU] = array(i, j, IU);
						state[IV] = array(i, j, IV);
						return state;
					}
				/**
				 * Set HydroState to global array (either conservative or primitive)
				 * at cell C(i, j) (local to global)
				 * @param[in, out]  array global array
				 * @param[in]       state HydroState of cell C(i, j)
				 * @param[in]       i     logical x-coordinate of the cell C
				 * @param[in]       j     logical y-coordinate of the cell C
				 */
				KOKKOS_INLINE_FUNCTION
					void setHydroState(DataArray array, const HydroState& state, int i, int j) const
					{
						array(i, j, ID) = state[ID];
						array(i, j, IP) = state[IP];
						array(i, j, IS) = state[IS];
						array(i, j, IU) = state[IU];
						array(i, j, IV) = state[IV];
					}
				KOKKOS_INLINE_FUNCTION
					HydroState  compute_GhostState(const int i, const int j, const HydroState& uLoc, const HydroState& uMx, const HydroState& uPx,\
							const HydroState& uMy,  const HydroState& uPy, const real_t dphix, const real_t dphiy) const
					{
						HydroState qghost;
						real_t phi =uLoc[IH];
						real_t p=ZERO_F, rho=ZERO_F, u=ZERO_F, v=ZERO_F, number = ZERO_F;
						const real_t g_x=params.settings.g_x;
						const real_t g_y=params.settings.g_y;
						const real_t dx=params.dx;
						const real_t dy=params.dy;
						int ghostWidth=params.ghostWidth;
						const bool barotropic = phi>ZERO_F? params.settings.barotropic1 : params.settings.barotropic0;

						if (i>ghostWidth)
						{
							const real_t phiMx=uMx[IH];
							if (phiMx*phi<=ZERO_F&&fmax(phiMx, phi)>ZERO_F)
							{
								HydroState qMx=computePrimitives(uMx);

								real_t rho0=qMx[ID], p0=qMx[IP];
								real_t delta_i=-1;
								const real_t num=ONE_F;
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
								const real_t vn    =qMx[IU]*dphix    +qMx[IV]*dphiy;
								const real_t vt    =qMx[IV]*dphix    -qMx[IU]*dphiy;

								rho+=rho0*num;
								p+=p0*num;
								u+=vn*num;
								v+=vt*num;
								number+=num;

							}
						}
						if (i<params.imax-ghostWidth)
						{
							const real_t phiPx=uPx[IH];
							if (phiPx*phi<=ZERO_F&&fmax(phiPx, phi)>ZERO_F)
							{
								HydroState qPx=computePrimitives(uPx);

								real_t rho0=qPx[ID], p0=qPx[IP];
								real_t delta_i=1;
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
								const real_t vn    =qPx[IU]*dphix    +qPx[IV]*dphiy;
								const real_t vt    =qPx[IV]*dphix    -qPx[IU]*dphiy;
								rho+=rho0*num;
								p+=p0*num;
								u+=vn*num;
								v+=vt*num;
								number+=num;
							}
						}
						if (j>ghostWidth)
						{
							const real_t phiMy=uMy[IH];
							if (phiMy*phi<=ZERO_F&&fmax(phiMy, phi)>ZERO_F)
							{
								HydroState qMy=computePrimitives(uMy);

								real_t rho0=qMy[ID], p0=qMy[IP];
								real_t delta_j=-1;
								const real_t num=ONE_F;
								const real_t dp=g_y*dy*delta_j;

								if (!barotropic)
								{
									p0-=dp*rho0;
								}
								else
								{
									const real_t pMy = computePressure(qMy);
									qMy[IP] = pMy - dp * rho0;
									rho0 = computeDensity(qMy);
								}
								const real_t vn    =qMy[IU]*dphix    +qMy[IV]*dphiy;
								const real_t vt    =qMy[IV]*dphix    -qMy[IU]*dphiy;
								rho+=rho0*num;
								u+=vn*num;
								v+=vt*num;
								p+=p0*num;
								number+=num;
							}
						}
						if (j<params.jmax-ghostWidth)
						{
							const real_t phiPy=uPy[IH];
							if (phiPy*phi<=ZERO_F&&fmax(phiPy, phi)>ZERO_F)
							{
								HydroState qPy=computePrimitives(uPy);

								real_t rho0=qPy[ID], p0=qPy[IP];
								real_t delta_j=1;
								const real_t num=ONE_F;
								const real_t dp=g_y*dy*delta_j;
								if (!barotropic)
								{
									p0-=dp*rho0;
								}
								else
								{
									const real_t pPy = computePressure(qPy);
									qPy[IP] = pPy - dp * rho0;
									rho0 = computeDensity(qPy);
								}
								const real_t vn    =qPy[IU]*dphix    +qPy[IV]*dphiy;
								const real_t vt    =qPy[IV]*dphix    -qPy[IU]*dphiy;

								rho+=rho0*num;
								u+=vn*num;
								v+=vt*num;
								p+=p0*num;
								number+=num;
							}
						}
						rho/=number;
						p/=number;
						u/=number;
						v/=number;
						const real_t ux=(dphix *      u   -   v    * dphiy) / (dphix * dphix + dphiy * dphiy);
						const real_t uy=(dphix *      v    +  u    * dphiy) / (dphix * dphix + dphiy * dphiy);
						qghost[IH]=phi>ZERO_F? -ONE_F:ONE_F;

						qghost[ID]=rho;
						qghost[IV]=uy;
						qghost[IU]=ux;
						qghost[IP]=p;

						return computeConservatives(qghost);
					}
				KOKKOS_INLINE_FUNCTION
					void computeAcousticRelaxationInter (const HydroState& qLoc, const HydroState& qNei, const real_t dphix,\
							const real_t dphiy, int delta_i, int delta_j, const real_t dpLoc, const real_t deltapsi, real_t& uStar, real_t& piStar) const
					{
						const real_t M=(fabs(qLoc[IH])*qLoc[ID]+fabs(qNei[IH])*qNei[ID])*deltapsi/fabs(fabs(qLoc[IH])+fabs(qNei[IH]));
						const real_t ratio=fabs(qLoc[IH])/fabs(qLoc[IH]-qNei[IH]);
						const real_t deltapLoc = ratio * deltapsi  * qLoc[ID]; 
						const real_t deltapNei = -(ONE_F - ratio)  * deltapsi * qNei[ID]; 
						const real_t deltap0   = -(HALF_F - ratio) * deltapsi * qLoc[ID]; 
						real_t gradphix, gradphiy, dir;
						if (params.settings.normal_direction)
						{
							gradphix = dphix;
							gradphiy = dphiy;
							dir   =qLoc[IH]>ZERO_F? -ONE_F: ONE_F;
						}
						else
						{
							gradphix = fabs(delta_i);
							gradphiy = fabs(delta_j);
							dir   =(delta_i>0||delta_j>0)? ONE_F:-ONE_F;
						}

						const real_t vn    =qLoc[IU]*gradphix    +qLoc[IV]*gradphiy;
						const real_t vn_Nei=qNei[IU]*gradphix    +qNei[IV]*gradphiy;
						const real_t vt    =qLoc[IV]*gradphix    -qLoc[IU]*gradphiy;

						const real_t cLoc= computeSpeedSound(qLoc);
						const real_t cNei= computeSpeedSound(qNei);

						real_t aLoc =params.settings.K * fmax(cLoc, cNei) * qLoc[ID];
						real_t aNei =params.settings.K * fmax(cLoc, cNei) * qNei[ID]; 
						const real_t pNei=computePressure(qNei);
						const real_t pLoc=computePressure(qLoc);



						real_t unStar = dir * (vn * aLoc + vn_Nei * aNei) / ( aLoc + aNei ) - (pNei - pLoc - dpLoc + M) / (aLoc + aNei);

						const real_t theta = params.settings.low_mach_correction ? FMIN(abs(uStar) / FMAX(cNei, cLoc), ONE_F) : ONE_F;
						const real_t ux=gradphix *  dir*     unStar -          vt    * gradphiy;
						const real_t uy=gradphix *           vt    +  dir  *   unStar * gradphiy;
						uStar = ux * delta_i + uy * delta_j;
						piStar = (aLoc * (pNei - deltapNei - dpLoc) + aNei * ( pLoc - deltapLoc))/(aLoc + aNei)\
							 +deltap0 - dir * theta  * aNei * aLoc * (vn_Nei - vn)/( aNei + aLoc);

					}
				KOKKOS_INLINE_FUNCTION
					void computeGradphiInter(const real_t gradphixLoc,const real_t gradphiyLoc,const real_t phiLoc,\
							const real_t gradphixNei,const real_t gradphiyNei,const real_t phiNei, real_t& dphix, real_t& dphiy)const
					{
						dphix=(fabs(phiNei)*gradphixLoc+fabs(phiLoc)*gradphixNei)/(fabs(phiNei)+fabs(phiLoc));
						dphiy=(fabs(phiNei)*gradphiyLoc+fabs(phiLoc)*gradphiyNei)/(fabs(phiNei)+fabs(phiLoc));

						const real_t dphiM=sqrt(dphix*dphix+dphiy*dphiy);
						dphix/=(dphiM);
						dphiy/=(dphiM);
					}
				KOKKOS_INLINE_FUNCTION
					void computeCurvatureInter(const real_t KLoc,const real_t  KNei, const real_t  phiLoc, const real_t phiNei, real_t& deltap)const
					{
						real_t Kinter=(fabs(phiLoc)*KNei+fabs(phiNei)*KLoc)/(fabs(phiLoc)+fabs(phiNei));
						Kinter=fabs(Kinter)>params.onesurdx?Kinter*params.onesurdx/fabs(Kinter):Kinter;
						deltap=(phiLoc<=ZERO_F)? params.settings.sigma*Kinter:-params.settings.sigma*Kinter;
					}


				/**
				 * Compute temperature using ideal gas law
				 * @param[in]  q  primitive variables array
				 * @param[out] T  temperature
				 */
				KOKKOS_INLINE_FUNCTION
					real_t computeDensity(const HydroState& q) const
					{
						const bool barotropic = q[IH]>ZERO_F? params.settings.barotropic0 : params.settings.barotropic1;
						if (!barotropic)
						{
							return q[ID];
						}
						else
						{
							const real_t rho0= q[IH]>ZERO_F? params.settings.rho0:params.settings.rho1;
							const real_t c= q[IH]>ZERO_F? params.settings.sound_speed0:params.settings.sound_speed1;
							const real_t p0= q[IH]>ZERO_F? params.Astate0:params.Astate1;
							return  rho0 + (q[IP] - p0)/c /c;
						}

					} // computeDensity
				KOKKOS_INLINE_FUNCTION
					real_t computeTemperature(const HydroState& q) const
					{
						const bool barotropic = q[IH]>ZERO_F? params.settings.barotropic0 : params.settings.barotropic1;
						if (!barotropic)
						{
							const real_t Rstar= q[IH]>ZERO_F? params.settings.Rstar0:params.settings.Rstar1;
							const real_t Bstate= q[IH]>ZERO_F? params.Bstate0:params.Bstate1;
							return (q[IP]+ Bstate) / (q[ID] * Rstar);
						}
						else
						{
							return q[IP];
						}

					} // computeTemperature
				KOKKOS_INLINE_FUNCTION
					real_t computePressure(const HydroState& q) const
					{
						const bool barotropic = q[IH]>ZERO_F? params.settings.barotropic0 : params.settings.barotropic1;
						if (!barotropic)
						{
							return q[IP];
						}
						else
						{
							const real_t rho0= q[IH]>ZERO_F? params.settings.rho0:params.settings.rho1;
							const real_t c= q[IH]>ZERO_F? params.settings.sound_speed0:params.settings.sound_speed1;
							const real_t p0= q[IH]>ZERO_F? params.Astate0:params.Astate1;
							return  p0 + (q[ID]- rho0) * c * c;
						}
					} // computePressure


				/**
				 * Compute speed of sound using ideal gas law
				 * @param[in]  q  primitive variables array
				 * @param[out] c  speed of sound
				 */
				KOKKOS_INLINE_FUNCTION
					void Compute_LocalValue(real_t xp1,real_t yp1,real_t& phi_L, real_t& dphi_dx, real_t& dphi_dy, DataArray U2data) const
					{
						const real_t dx=params.dx;
						const real_t dy=params.dy;
						const real_t xmin=params.xmin;
						const real_t xmax=params.xmax;
						const real_t ymin=params.ymin;
						const real_t ymax=params.ymax;
						const int    ghostWidth=params.ghostWidth;

						if ((xp1>xmin+dx)&&(xp1<xmax-dx)&&(yp1>ymin+dy)&&(yp1<ymax-dy))
						{
							int i=(xp1-xmin-HALF_F*dx)/dx+ghostWidth;
							int j=(yp1-ymin-HALF_F*dy)/dy+ghostWidth;

							const double a00 =     U2data(i, j, IH);

							const double a01 = -.5*U2data(i, j-1, IH)   + .5 *U2data(i, j+1, IH);

							const double a02 =     U2data(i, j-1, IH)   - 2.5*U2data(i, j, IH)         + 2  *U2data(i, j+1, IH)   -  .5*U2data(i, j+2, IH);

							const double a03 = -.5*U2data(i, j-1, IH)   + 1.5*U2data(i, j, IH)         - 1.5*U2data(i, j+1, IH)   +  .5*U2data(i, j+2, IH);

							const double a10 = -.5*U2data(i-1, j, IH) +  .5*U2data(i+1, j, IH);

							const double a11 = .25*U2data(i-1, j-1, IH) - .25*U2data(i-1, j+1, IH) - .25*U2data(i+1, j-1, IH) + .25*U2data(i+1, j+1, IH);

							const double a12 = -.5*U2data(i-1, j-1, IH) +1.25*U2data(i-1, j, IH)       - U2data(i-1, j+1, IH) + .25*U2data(i-1, j+2, IH)\
									   +.5*U2data(i+1, j-1, IH) -1.25*U2data(i+1, j, IH)       + U2data(i+1, j+1, IH) - .25*U2data(i+1, j+2, IH);

							const double a13 = .25*U2data(i-1, j-1, IH) - .75*U2data(i-1, j, IH)       + .75*U2data(i-1, j+1, IH) - .25*U2data(i-1, j+2, IH) \
									   -.25*U2data(i+1, j-1, IH) + .75*U2data(i+1, j, IH)       - .75*U2data(i+1, j+1, IH) + .25*U2data(i+1, j+2, IH) ;

							const double a20 =     U2data(i-1, j, IH)   - 2.5*U2data(i, j, IH)         +  2.*U2data(i+1, j, IH)   -  .5*U2data(i+2, j, IH);

							const double a21 = -.5*U2data(i-1, j-1, IH) +  .5*U2data(i-1, j+1, IH) +1.25*U2data(i, j-1, IH)   -1.25*U2data(i, j+1, IH) \
									   -   U2data(i+1, j-1, IH) +     U2data(i+1, j+1, IH) + .25*U2data(i+2, j-1, IH) - .25*U2data(i+2, j+1, IH);

							const double a22 =     U2data(i-1, j-1, IH) - 2.5*U2data(i-1, j, IH)       +   2*U2data(i-1, j+1, IH) -  .5*U2data(i-1, j+2, IH)\
									       - 2.5*U2data(i,   j-1, IH) +6.25*U2data(i,   j, IH)         - 5*U2data(i,   j+1, IH) +1.25*U2data(i  , j+2, IH) \
									       +   2*U2data(i+1, j-1, IH) -   5*U2data(i+1, j, IH)       +   4*U2data(i+1, j+1, IH) -     U2data(i+1, j+2, IH) \
									       -  .5*U2data(i+2, j-1, IH) +1.25*U2data(i+2, j, IH)       -     U2data(i+2, j+1, IH) + .25*U2data(i+2, j+2, IH);


							const double a23 = -.5*U2data(i-1, j-1, IH) + 1.5*U2data(i-1, j, IH)       - 1.5*U2data(i-1, j+1, IH) +  .5*U2data(i-1, j+2, IH) \
									   + 1.25*U2data(i,   j-1, IH) -3.75*U2data(i,   j, IH)       +3.75*U2data(i,   j+1, IH) -1.25*U2data(i,   j+2, IH) \
									   -   U2data(i+1, j-1, IH) +   3*U2data(i+1, j, IH)       -   3*U2data(i+1, j+1, IH) +     U2data(i+1, j+2, IH)\
									   +  .25*U2data(i+2, j-1, IH) - .75*U2data(i+2, j, IH)       + .75*U2data(i+2, j+1, IH) - .25*U2data(i+2, j+2, IH);

							const double a30 = -.5*U2data(i-1, j, IH)       + 1.5*U2data(i, j, IH)         - 1.5*U2data(i+1, j, IH)       +  .5*U2data(i+2, j, IH);

							const double a31 = .25*U2data(i-1, j-1, IH)     - .25*U2data(i-1, j+1, IH) - .75*U2data(i  , j-1, IH)   + .75*U2data(i  , j+1, IH) \
									   + .75*U2data(i+1, j-1, IH)     - .75*U2data(i+1, j+1, IH) - .25*U2data(i+2, j-1, IH)   + .25*U2data(i+2, j+1, IH);

							const double a32 = -.5*U2data(i-1, j-1, IH) +1.25*U2data(i-1, j, IH)       -     U2data(i-1, j+1, IH) + .25*U2data(i-1, j+2, IH)\
									   + 1.5*U2data(i  , j-1, IH) -3.75*U2data(i  , j, IH)       +   3*U2data(i  , j+1, IH) - .75*U2data(i  , j+2, IH)\
									   - 1.5*U2data(i+1, j-1, IH) +3.75*U2data(i+1, j, IH)       -   3*U2data(i+1, j+1, IH) + .75*U2data(i+1, j+2, IH)\
									   +  .5*U2data(i+2, j-1, IH) -1.25*U2data(i+2, j, IH)       +     U2data(i+2, j+1, IH) - .25*U2data(i+2, j+2, IH);

							const double a33 = .25*U2data(i-1, j-1, IH) - .75*U2data(i-1, j, IH)       + .75*U2data(i-1, j+1, IH) - .25*U2data(i-1, j+2, IH)\
									   - .75*U2data(i  , j-1, IH) +2.25*U2data(i  , j, IH)       -2.25*U2data(i  , j+1, IH) + .75*U2data(i  , j+2, IH)\
									   + .75*U2data(i+1, j-1, IH) -2.25*U2data(i+1, j, IH)       +2.25*U2data(i+1, j+1, IH) - .75*U2data(i+1, j+2, IH)\
									   - .25*U2data(i+2, j-1, IH) + .75*U2data(i+2, j, IH)       - .75*U2data(i+2, j+1, IH) + .25*U2data(i+2, j+2, IH);


							const double xabs=(xp1-xmin-HALF_F*dx)/dx-i+ghostWidth;
							const double yabs=(yp1-ymin-HALF_F*dy)/dy-j+ghostWidth;
							const double xabs2=xabs*xabs;
							const double xabs3=xabs2*xabs;
							const double yabs2=yabs*yabs;
							const double yabs3=yabs2*yabs;

							phi_L=  (a00+a01*yabs+a02*yabs2+a03*yabs3)+
								(a10+a11*yabs+a12*yabs2+a13*yabs3)*xabs+
								(a20+a21*yabs+a22*yabs2+a23*yabs3)*xabs2+
								(a30+a31*yabs+a32*yabs2+a33*yabs3)*xabs3;

							dphi_dx=       ((a10+a11*yabs+a12*yabs2+a13*yabs3)+
									(a20+a21*yabs+a22*yabs2+a23*yabs3)*2*xabs+
									(a30+a31*yabs+a32*yabs2+a33*yabs3)*3*xabs2)/dx;

							dphi_dy=       ((a01+2.*a02*yabs+3.*a03*yabs2)+
									(a11+2.*a12*yabs+3.*a13*yabs2)*xabs+
									(a21+2.*a22*yabs+3.*a23*yabs2)*xabs2+
									(a31+2.*a32*yabs+3.*a33*yabs2)*xabs3)/dy;

						}
						else
						{

							int i=int((xp1-xmin-HALF_F*dx)/dx)+ghostWidth;
							int j=int((yp1-ymin-HALF_F*dy)/dy)+ghostWidth;
							if (i==params.nx+ghostWidth-1)
								i-=1;
							if (j==params.ny+ghostWidth-1)
								j-=1;
							real_t xL1=xmin+(i-ghostWidth+HALF_F)*dx;
							real_t yL1=ymin+(j-ghostWidth+HALF_F)*dy;

							phi_L=          ((yL1+dy-yp1)*(xL1+dx-xp1)*U2data(i, j, IH)+\
									(yL1+dy-yp1)*(xp1-xL1)  *U2data(i+1, j, IH)+\
									(yp1-yL1)  *(xL1+dx-xp1)*+U2data(i, j+1, IH)+\
									(yp1-yL1)  *(xp1-xL1)  *U2data(i+1, j+1, IH))/(dx*dy); 
							dphi_dx=((U2data(i+1, j, IH) -U2data(i, j, IH) )*(yL1+dx-yp1)\
									+(U2data(i+1, j+1, IH)-U2data(i, j+1, IH))*(yp1-yL1))/(dx*dy);

							dphi_dy=((U2data(i, j+1, IH) -U2data(i, j, IH))  *(xL1+dx-xp1)\
									+(U2data(i+1, j+1, IH)-U2data(i+1, j, IH)) *(xp1-xL1))/(dx*dy);
						}

					}
				KOKKOS_INLINE_FUNCTION
					real_t Compute_height_x(int i, int j, int del, DataArray U2data) const
					{
						const real_t tolerance=.0000000000001;
						const real_t dx=params.dx;
						const real_t dy=params.dy;
						const int ghostWidth=params.ghostWidth;

						real_t xp1, phip1;

						real_t y1=params.ymin+(j+del+0.5-ghostWidth)*dy;
						real_t x1=params.xmin+(i+0.5-ghostWidth)*dx;
						real_t phio=U2data(i, j+del, IH);
						phip1=phio;

						xp1=x1;
						int n=0;
						real_t dphix, dphiy;
						while (fabs(phip1)>tolerance&&n<20)
						{
							Compute_LocalValue(xp1, y1,phip1, dphix, dphiy, U2data);

							xp1-=0.8*dphix*phip1/(sqrt(dphix*dphix));

							n++;
						}
						return xp1-x1;
					}
				KOKKOS_INLINE_FUNCTION
					real_t Compute_height_y(int i, int j, int del, DataArray U2data) const
					{
						const real_t tolerance=.0000000000001;
						const real_t dx=params.dx;
						const real_t dy=params.dy;
						const int ghostWidth=params.ghostWidth;

						real_t yp1, phip1;

						real_t y1=params.ymin+(j+0.5-ghostWidth)*dy;
						real_t x1=params.xmin+(i+del+0.5-ghostWidth)*dx;
						real_t phio=U2data(i+del, j, IH);
						phip1=phio;

						yp1=y1;
						int n=0;
						real_t dphix, dphiy;
						while (fabs(phip1)>tolerance&&n<30)
						{
							Compute_LocalValue(x1, yp1,phip1, dphix, dphiy, U2data);

							yp1-=dphiy*phip1/(sqrt(dphiy*dphiy));

							n++;
						}
						return yp1-y1;
					}
				KOKKOS_INLINE_FUNCTION
					real_t computeSpeedSound(const HydroState& q) const
					{
						const real_t gamma=(q[IH]>ZERO_F)?  params.settings.gamma0 :params.settings.gamma1;
						const real_t Bstate=(q[IH]>ZERO_F)? params.Bstate0:params.Bstate1;
						const bool barotropic=(q[IH]>ZERO_F)?params.settings.barotropic0:params.settings.barotropic1;

						if (!barotropic)
							return SQRT(gamma * (q[IP]+Bstate) / q[ID]);
						else
						{
							const real_t c = q[IH] > ZERO_F ? params.settings.sound_speed0 : params.settings.sound_speed1;
							return c;
						}

					} // computeSpeedSound

				/**
				 * Convert conservative variables (rho, rho*u, rho*v, rho*E) to
				 * primitive variables (rho, u, v, p) using ideal gas law
				 * @param[in]  u  conservative variables array
				 * @param[out] q  primitive    variables array
				 */
				KOKKOS_INLINE_FUNCTION
					HydroState computePrimitives(const HydroState& u) const
					{
						const real_t phiLoc=u[IH];
						const real_t gamma=(phiLoc>ZERO_F)?params.settings.gamma0:params.settings.gamma1;
						const real_t Bstate=(phiLoc>ZERO_F)?params.Bstate0:params.Bstate1;
						const bool barotropic=(phiLoc>ZERO_F)?params.settings.barotropic0:params.settings.barotropic1;


						const real_t invD = ONE_F / u[ID];
						const real_t ekin = HALF_F*(u[IU]*u[IU]+u[IV]*u[IV])*invD;

						HydroState q;
						q[ID] = u[ID];
						q[IH] = u[IH];
						q[IU] = u[IU] * invD;
						q[IV] = u[IV] * invD;
						if (!barotropic)
							q[IP] = (gamma-ONE_F) * (u[IE] - ekin)-gamma* Bstate;
						else
							q[IP] = u[IE];


						return q;
					} // computePrimitives

				/**
				 * Convert primitive variables (rho, p, u, v) to
				 * conservative variables (rho, rho*E, rho*u, rho*v) using ideal gas law
				 * @param[in]  q  primitive    variables array
				 * @param[out] u  conservative variables array
				 */
				KOKKOS_INLINE_FUNCTION
					HydroState computeConservatives(const HydroState& q) const
					{
						const real_t phiLoc=q[IH];
						const real_t gamma=(phiLoc>ZERO_F)?params.settings.gamma0:params.settings.gamma1;
						const real_t Bstate=(phiLoc>ZERO_F)?params.Bstate0:params.Bstate1;
						const bool  barotropic=(phiLoc>ZERO_F)?params.settings.barotropic0:params.settings.barotropic1;


						const real_t ekin = HALF_F*q[ID]*(q[IU]*q[IU]+q[IV]*q[IV]);

						HydroState u;
						u[ID] = q[ID];
						u[IH] = q[IH];
						u[IU] = q[ID] * q[IU];
						u[IV] = q[ID] * q[IV];
						if (!barotropic)
							u[IE] = (q[IP]+gamma*Bstate)/(gamma-ONE_F) + ekin;
						else
							u[IE] = q[IP];


						return u;
					} // computeConservatives
		}; // class HydroBaseFunctor2D

	} // namespace all_regime

} // namespace euler_kokkos
