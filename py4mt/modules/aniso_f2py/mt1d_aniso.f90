!*==mt1d_aniso.f90 processed by SPAG 8.04RA 21:21  7 Nov 2025
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
SUBROUTINE mt1d_aniso(layani,h,al,at,blt,nl,per,z,dzdal,dzdat,dzdbs,dzdh)
 
!f2py intent(in)      layani,h,al,at,blt,nl,per
!f2py depend(nl)      layani,h,al,at,blt
!f2py intent (out)    z,dzdal,dzdat,dzdbs,dzdh
!f2py threadsafe
 
   USE iso_fortran_env
   USE ISO_FORTRAN_ENV                 
   IMPLICIT NONE
!
! PARAMETER definitions rewritten by SPAG
!
   REAL(kind(1.0D0)) , PARAMETER :: pi = 3.14159265358979323846264338327950288D0
   COMPLEX(kind(1.0D0)) , PARAMETER :: ic = (0.D0,1.D0)
   INTEGER , PARAMETER :: nlmax = 1001
!
! Dummy argument declarations rewritten by SPAG
!
   INTEGER(int32) , INTENT(INOUT) , DIMENSION(nlmax) :: layani
   REAL(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax) :: h
   REAL(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax) :: al
   REAL(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax) :: at
   REAL(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax) :: blt
   INTEGER :: nl
   REAL(kind(1.0D0)) , INTENT(IN) :: per
   COMPLEX(kind(1.0D0)) , DIMENSION(2,2) :: z
   COMPLEX(kind(1.0D0)) , INTENT(INOUT) , DIMENSION(nlmax,2,2) :: dzdal
   COMPLEX(kind(1.0D0)) , INTENT(INOUT) , DIMENSION(nlmax,2,2) :: dzdat
   COMPLEX(kind(1.0D0)) , INTENT(INOUT) , DIMENSION(nlmax,2,2) :: dzdbs
   COMPLEX(kind(1.0D0)) , INTENT(INOUT) , DIMENSION(nlmax,2,2) :: dzdh
!
! Local variable declarations rewritten by SPAG
!
   REAL(kind(1.0D0)) :: a1 , a1is , a2 , a2is , bs , bsref , c2bs , hd , mu0 , omega , s2bs
   COMPLEX(kind(1.0D0)) :: ag1 , ag2 , dtzbot , dz1 , dz2 , iom , k0 , k1 , k2 , zdenom
   COMPLEX(kind(1.0D0)) , EXTERNAL :: dfm , dfp
   COMPLEX(kind(1.0D0)) , DIMENSION(2,2) :: dzbot , dztop , zbot , zprd , zrot
   COMPLEX(kind(1.0D0)) , DIMENSION(nlmax,2,2) :: dzdalrot , dzdatrot , dzdbsrot , dzdhrot
   INTEGER :: il , ix , iy , layer , layer1 , nl1
   EXTERNAL rotz , rotzs , zscua1 , zscua2 , zscubs , zscuh , zsprpg
!
! End of declarations rewritten by SPAG
!
!
! PARAMETER definitions rewritten by SPAG
!
!
! Dummy argument declarations rewritten by SPAG
!
!
! Local variable declarations rewritten by SPAG
!
!
! End of declarations rewritten by SPAG
!
!     =============================================
!     Stable impedance propagation procedure for 1-D layered
!     media with generally anisotropic layers. Computes both
!     the impedances and parametric sensitivities, the latter
!     only with respect to the effective horizontal anisotropy
!     parameters of the layer, i.e., longitudinal conductivity
!     AL (A1), transversal conductivity AT (A2), and the azi-
!     muthal anisotropy strike BLT (BS). For isotropic layers,
!     the sensitivity with respect to the common conductivity
!     can be evaluated as well. Additionally, the sensitivity
!     with respect to the thickesses of the layers is computed.
!
!     Input:
!     ------
!     LAYANI(NLMAX)....integer*4 array of 0/1 indices that
!                      are set to 0 for layers to be treated
!                      consequently as isotropic. For layers
!                      with different princial conductivities
!                      the isotropy flag is changed automatically
!                      to 1 (i.e., anisotropic layer)
!     H(NLMAX).........real*8 array of layer thicknesses in km.
!                      Any value may be input as a thickness of
!                      the homogeneous basement
!     AL(NLMAX)........real*8 array of maximum horizontal con-
!                      ductivities of the layers in S/m
!     AT(NLMAX)........real*8 array of minimum horizontal con-
!                      ductivities of the layers in S/m
!     BLT(NLMAX).......real*8 array of equivalent horizontal
!                      anisotropy strikes of the layers in
!                      radians
!     NL...............integer*4 number of layers in the model
!                      including the basement, NL<=NLMAX
!     PER..............real*8 period of the elmg field in s
!
!     Output:
!     -------
!     Z(2,2)...........complex*16 array with the elements of the
!                      impedance tensor on the surface in SI units
!                      (Ohm). The arrangement is as follows
!                         | Z_xx  Z_xy | = | Z(1,1)  Z(1,2) |
!                         | Z_yx  Z_yy | = | Z(2,1)  Z(2,2) |
!     DZDAL(NLMAX,2,2).complex*16 array with partial derivatives of
!                      the impedance Z with respect to the maximum
!                      horizontal conductivity of the layers,
!                           DZDAL(I,J,K)=d{Z(J,K)}/d{AL(I)},
!                      in SI units (Ohm**2)
!     DZDAT(NLMAX,2,2).complex*16 array with partial derivatives of
!                      the impedance Z with respect to the minimum
!                      horizontal conductivity of the layers,
!                           DZDAT(I,J,K)=d{Z(J,K)}/d{AT(I)},
!                      in SI units (Ohm*2)
!     DZDBS(NLMAX,2,2).complex*16 array with partial derivatives of
!                      the impedance Z with respect to the effective
!                      horizontal anisotropy strike of the layers,
!                           DZDBS(I,J,K)=d{Z(J,K)}/d{BLT(I)},
!                      in SI units (Ohm/radian=Ohm)
!     DZDH(NLMAX,2,2)..complex*16 array with partial derivatives of
!                      the impedance Z with respect to the thickness
!                      of the layers
!                           DZDH(I,J,K)=d{Z(J,K)}/d{H(I)},
!                      in SI units (Ohm/m!!!), NOT in Ohm/km!!!
!
!
!
!
!
!
   omega = 2.D0*pi/dble(per)
   mu0 = 4.D-7*pi
   iom = -ic*omega*mu0
   k0 = (1.D0-ic)*2.D-3*pi/dsqrt(10.D0*dble(per))
!
!> Compute the impedance on the top of the homogeneous
!> basement in the direction of its strike
!
   layer = nl
   a1 = al(layer)
   a2 = at(layer)
   bs = blt(layer)
!
   k1 = k0*dsqrt(a1)
   k2 = k0*dsqrt(a2)
   c2bs = dcos(2.D0*bs)
   s2bs = dsin(2.D0*bs)
   a1is = 1.D0/dsqrt(a1)
   a2is = 1.D0/dsqrt(a2)
!
   zrot(1,1) = 0.D0
   zrot(1,2) = k0*a1is
   zrot(2,1) = -k0*a2is
   zrot(2,2) = 0.D0
   CALL rotz(zrot,-bs,zprd)
!
!> In the isotropic case, compute the sensitivity with
!> respect to the isotropic basement conductivity
!
   IF ( layani(layer)==0 .AND. a1==a2 ) THEN
      dzdalrot(layer,1,1) = 0.D0
      dzdalrot(layer,1,2) = -0.5D0*zrot(1,2)/a1
      dzdalrot(layer,2,1) = 0.D0
      dzdalrot(layer,2,2) = 0.D0
!
      dzdatrot(layer,1,1) = 0.D0
      dzdatrot(layer,1,2) = 0.D0
      dzdatrot(layer,2,1) = -0.5D0*zrot(2,1)/a2
      dzdatrot(layer,2,2) = 0.D0
!
      dzdbsrot(layer,1,1) = 0.D0
      dzdbsrot(layer,1,2) = 0.D0
      dzdbsrot(layer,2,1) = 0.D0
      dzdbsrot(layer,2,2) = 0.D0
!
!> For the anisotropic case, compute the sensitivies
!> separately with respect to the individual principal
!> conductivities and with respect to the effective
!> anisotropy strike of the basement
!
   ELSE
      dzdalrot(layer,1,1) = 0.D0
      dzdalrot(layer,1,2) = -0.5D0*zrot(1,2)/a1
      dzdalrot(layer,2,1) = 0.D0
      dzdalrot(layer,2,2) = 0.D0
!
      dzdatrot(layer,1,1) = 0.D0
      dzdatrot(layer,1,2) = 0.D0
      dzdatrot(layer,2,1) = -0.5D0*zrot(2,1)/a2
      dzdatrot(layer,2,2) = 0.D0
!
      dzdbsrot(layer,1,1) = -zrot(1,2) - zrot(2,1)
      dzdbsrot(layer,1,2) = 0.D0
      dzdbsrot(layer,2,1) = 0.D0
      dzdbsrot(layer,2,2) = -dzdbsrot(layer,1,1)
      layani(layer) = 1
   ENDIF
!
!> Sensitivity of the basement impedance with respect to
!> the thickness is always zero
!
   dzdhrot(layer,1,1) = 0.D0
   dzdhrot(layer,1,2) = 0.D0
   dzdhrot(layer,2,1) = 0.D0
   dzdhrot(layer,2,2) = 0.D0
!
!> If no more layers are present in the model, rotate the
!> impedance and the sensitivities (except the one with respect
!> to the thickness) back into the original coordinate system
!> and return
!
   IF ( nl==1 ) THEN
      CALL rotz(zrot,-bs,z)
      CALL rotzs(dzdalrot,nl,layer,-bs,dzdal)
      CALL rotzs(dzdatrot,nl,layer,-bs,dzdat)
      CALL rotzs(dzdbsrot,nl,layer,-bs,dzdbs)
      dzdh(layer,1,1) = 0.D0
      dzdh(layer,1,2) = 0.D0
      dzdh(layer,2,1) = 0.D0
      dzdh(layer,2,2) = 0.D0
      RETURN
   ENDIF
!
!> Set the reference direction to the anisotropy strike of
!> the current layer
!
   bsref = bs
!
!> Process the rest of the layers above the basement
!
   nl1 = nl - 1
   DO layer = nl1 , 1 , -1
      hd = 1.D+3*dble(h(layer))
      a1 = al(layer)
      a2 = at(layer)
      bs = blt(layer)
!
!> If the strike direction differs from that of the previous
!> layer, rotate the impedance and the parametric sensitivities
!> of all deeper layers into the coordinate system of the
!> current anisotropy strike. Store the rotated sensitivities
!> in the positions of the original sensitivity variables
!
      layer1 = layer + 1
      dtzbot = zrot(1,1)*zrot(2,2) - zrot(1,2)*zrot(2,1)
      IF ( bs/=bsref .AND. a1/=a2 ) THEN
         CALL rotz(zrot,bs-bsref,zbot)
         CALL rotzs(dzdalrot,nl,layer1,bs-bsref,dzdal)
         CALL rotzs(dzdatrot,nl,layer1,bs-bsref,dzdat)
         CALL rotzs(dzdbsrot,nl,layer1,bs-bsref,dzdbs)
         CALL rotzs(dzdhrot,nl,layer1,bs-bsref,dzdh)
!
!> If the anisotropy strike does not change, or for the isotropic
!> case, take the impedances from the top of the layer immediately
!> below as the bottom impedance for the current layer without any
!> change. The same applies to the sensitivities of the deeper
!> layers.
!
      ELSE
         zbot(1,1) = zrot(1,1)
         zbot(1,2) = zrot(1,2)
         zbot(2,1) = zrot(2,1)
         zbot(2,2) = zrot(2,2)
         bs = bsref
         DO ix = 1 , 2
            DO iy = 1 , 2
               DO il = layer1 , nl
                  dzdal(il,ix,iy) = dzdalrot(il,ix,iy)
                  dzdat(il,ix,iy) = dzdatrot(il,ix,iy)
                  dzdbs(il,ix,iy) = dzdbsrot(il,ix,iy)
                  dzdh(il,ix,iy) = dzdhrot(il,ix,iy)
               ENDDO
            ENDDO
         ENDDO
      ENDIF
!
      k1 = k0*dsqrt(a1)
      k2 = k0*dsqrt(a2)
      a1is = 1.D0/dsqrt(a1)
      a2is = 1.D0/dsqrt(a2)
      dz1 = k0*a1is
      dz2 = k0*a2is
      ag1 = k1*hd
      ag2 = k2*hd
!
!> Propagate the impedance tensor from the bottom to the top
!> of the current layer
!
      zdenom = dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + zbot(1,2)*dfm(ag1)*dfp(ag2)/dz1 - zbot(2,1)*dfp(ag1)*dfm(ag2)/dz2 + dfp(ag1)   &
             & *dfp(ag2)
      zrot(1,1) = 4.D0*zbot(1,1)*cdexp(-ag1-ag2)/zdenom
      zrot(1,2) = (zbot(1,2)*dfp(ag1)*dfp(ag2)-zbot(2,1)*dfm(ag1)*dfm(ag2)*dz1/dz2+dtzbot*dfp(ag1)*dfm(ag2)/dz2+dfm(ag1)*dfp(ag2)  &
                & *dz1)/zdenom
      zrot(2,1) = (zbot(2,1)*dfp(ag1)*dfp(ag2)-zbot(1,2)*dfm(ag1)*dfm(ag2)*dz2/dz1-dtzbot*dfm(ag1)*dfp(ag2)/dz1-dfp(ag1)*dfm(ag2)  &
                & *dz2)/zdenom
      zrot(2,2) = 4.D0*zbot(2,2)*cdexp(-ag1-ag2)/zdenom
      CALL rotz(zrot,-bs,zprd)
!
!> Propagate all the parametric sensitivities of the deeper
!> than the current layer from the bottom to the top of the
!> current layer
!
      DO il = layer1 , nl
         dzbot(1,1) = dzdal(il,1,1)
         dzbot(1,2) = dzdal(il,1,2)
         dzbot(2,1) = dzdal(il,2,1)
         dzbot(2,2) = dzdal(il,2,2)
         CALL zsprpg(dzbot,zbot,zrot,dz1,dz2,ag1,ag2,dztop)
         dzdalrot(il,1,1) = dztop(1,1)
         dzdalrot(il,1,2) = dztop(1,2)
         dzdalrot(il,2,1) = dztop(2,1)
         dzdalrot(il,2,2) = dztop(2,2)
!
         dzbot(1,1) = dzdat(il,1,1)
         dzbot(1,2) = dzdat(il,1,2)
         dzbot(2,1) = dzdat(il,2,1)
         dzbot(2,2) = dzdat(il,2,2)
         CALL zsprpg(dzbot,zbot,zrot,dz1,dz2,ag1,ag2,dztop)
         dzdatrot(il,1,1) = dztop(1,1)
         dzdatrot(il,1,2) = dztop(1,2)
         dzdatrot(il,2,1) = dztop(2,1)
         dzdatrot(il,2,2) = dztop(2,2)
!
         dzbot(1,1) = dzdbs(il,1,1)
         dzbot(1,2) = dzdbs(il,1,2)
         dzbot(2,1) = dzdbs(il,2,1)
         dzbot(2,2) = dzdbs(il,2,2)
         CALL zsprpg(dzbot,zbot,zrot,dz1,dz2,ag1,ag2,dztop)
         dzdbsrot(il,1,1) = dztop(1,1)
         dzdbsrot(il,1,2) = dztop(1,2)
         dzdbsrot(il,2,1) = dztop(2,1)
         dzdbsrot(il,2,2) = dztop(2,2)
!
         dzbot(1,1) = dzdh(il,1,1)
         dzbot(1,2) = dzdh(il,1,2)
         dzbot(2,1) = dzdh(il,2,1)
         dzbot(2,2) = dzdh(il,2,2)
         CALL zsprpg(dzbot,zbot,zrot,dz1,dz2,ag1,ag2,dztop)
         dzdhrot(il,1,1) = dztop(1,1)
         dzdhrot(il,1,2) = dztop(1,2)
         dzdhrot(il,2,1) = dztop(2,1)
         dzdhrot(il,2,2) = dztop(2,2)
      ENDDO
!
!> Compute the parametric sensitivities with respect to the
!> parameters of the current layer
!
!> For the isotropic case
!
      IF ( layani(layer)==0 .AND. a1==a2 ) THEN
         CALL zscua1(zbot,zrot,a1,dz1,dz2,ag1,ag2,dztop)
         dzdalrot(layer,1,1) = dztop(1,1)
         dzdalrot(layer,1,2) = dztop(1,2)
         dzdalrot(layer,2,1) = dztop(2,1)
         dzdalrot(layer,2,2) = dztop(2,2)
!
         CALL zscua2(zbot,zrot,a2,dz1,dz2,ag1,ag2,dztop)
         dzdatrot(layer,1,1) = dztop(1,1)
         dzdatrot(layer,1,2) = dztop(1,2)
         dzdatrot(layer,2,1) = dztop(2,1)
         dzdatrot(layer,2,2) = dztop(2,2)
!
         dzdbsrot(layer,1,1) = 0.D0
         dzdbsrot(layer,1,2) = 0.D0
         dzdbsrot(layer,2,1) = 0.D0
         dzdbsrot(layer,2,2) = 0.D0
!
         CALL zscuh(zbot,zrot,a1,a2,dz1,dz2,ag1,ag2,dztop)
         dzdhrot(layer,1,1) = dztop(1,1)
         dzdhrot(layer,1,2) = dztop(1,2)
         dzdhrot(layer,2,1) = dztop(2,1)
         dzdhrot(layer,2,2) = dztop(2,2)
!
!> For the general anisotropic case
!
      ELSE
         CALL zscua1(zbot,zrot,a1,dz1,dz2,ag1,ag2,dztop)
         dzdalrot(layer,1,1) = dztop(1,1)
         dzdalrot(layer,1,2) = dztop(1,2)
         dzdalrot(layer,2,1) = dztop(2,1)
         dzdalrot(layer,2,2) = dztop(2,2)
!
         CALL zscua2(zbot,zrot,a2,dz1,dz2,ag1,ag2,dztop)
         dzdatrot(layer,1,1) = dztop(1,1)
         dzdatrot(layer,1,2) = dztop(1,2)
         dzdatrot(layer,2,1) = dztop(2,1)
         dzdatrot(layer,2,2) = dztop(2,2)
!
         CALL zscubs(zbot,zrot,dz1,dz2,ag1,ag2,dztop)
         dzdbsrot(layer,1,1) = dztop(1,1)
         dzdbsrot(layer,1,2) = dztop(1,2)
         dzdbsrot(layer,2,1) = dztop(2,1)
         dzdbsrot(layer,2,2) = dztop(2,2)
!
         CALL zscuh(zbot,zrot,a1,a2,dz1,dz2,ag1,ag2,dztop)
         dzdhrot(layer,1,1) = dztop(1,1)
         dzdhrot(layer,1,2) = dztop(1,2)
         dzdhrot(layer,2,1) = dztop(2,1)
         dzdhrot(layer,2,2) = dztop(2,2)
!
         layani(layer) = 1
      ENDIF
!
!> Set the reference direction to the anisotropy strike of
!> the current layer and go on to the next layer
!
      bsref = bs
!
   ENDDO
!
!> On the surface, rotate both the impedance and the parametric
!> sensitivities into the original coordinate system and return
!
   IF ( bsref/=0.D0 ) THEN
      CALL rotz(zrot,-bsref,z)
      CALL rotzs(dzdalrot,nl,1,-bsref,dzdal)
      CALL rotzs(dzdatrot,nl,1,-bsref,dzdat)
      CALL rotzs(dzdbsrot,nl,1,-bsref,dzdbs)
      CALL rotzs(dzdhrot,nl,1,-bsref,dzdh)
   ELSE
      z(1,1) = zrot(1,1)
      z(1,2) = zrot(1,2)
      z(2,1) = zrot(2,1)
      z(2,2) = zrot(2,2)
      DO ix = 1 , 2
         DO iy = 1 , 2
            DO il = 1 , nl
               dzdal(il,ix,iy) = dzdalrot(il,ix,iy)
               dzdat(il,ix,iy) = dzdatrot(il,ix,iy)
               dzdbs(il,ix,iy) = dzdbsrot(il,ix,iy)
               dzdh(il,ix,iy) = dzdhrot(il,ix,iy)
            ENDDO
         ENDDO
      ENDDO
   ENDIF
!
!
END SUBROUTINE mt1d_aniso
!*==zs1anis.f90 processed by SPAG 8.04RA 21:21  7 Nov 2025
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!===> END SUBROUTINE ZS1ANEF
!
!
!===> SUBROUTINE ZS1ANIS
!     ================================================================
SUBROUTINE zs1anis(layani,h,rop,ustr,udip,usla,al,at,blt,nl,per,z,dzdal,dzdat,dzdbs,dzdh,dzdsgpx,dzdsgpy,dzdsgpz,dzdstr,dzddip,    &
                 & dzdsla)
   USE iso_fortran_env
   USE ISO_FORTRAN_ENV                 
   IMPLICIT NONE
!
! PARAMETER definitions rewritten by SPAG
!
   REAL(kind(1.0D0)) , PARAMETER :: pi = 3.14159265358979323846264338327950288D0
   INTEGER , PARAMETER :: nlmax = 1001
!
! Dummy argument declarations rewritten by SPAG
!
   INTEGER(int32) , DIMENSION(nlmax) :: layani
   REAL(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax) :: h
   REAL(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax,3) :: rop
   REAL(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax) :: ustr
   REAL(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax) :: udip
   REAL(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax) :: usla
   REAL(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax) :: al
   REAL(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax) :: at
   REAL(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax) :: blt
   INTEGER , INTENT(IN) :: nl
   REAL(kind(1.0D0)) :: per
   COMPLEX(kind(1.0D0)) , DIMENSION(2,2) :: z
   COMPLEX(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax,2,2) :: dzdal
   COMPLEX(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax,2,2) :: dzdat
   COMPLEX(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax,2,2) :: dzdbs
   COMPLEX(kind(1.0D0)) , DIMENSION(nlmax,2,2) :: dzdh
   COMPLEX(kind(1.0D0)) , INTENT(OUT) , DIMENSION(nlmax,2,2) :: dzdsgpx
   COMPLEX(kind(1.0D0)) , INTENT(OUT) , DIMENSION(nlmax,2,2) :: dzdsgpy
   COMPLEX(kind(1.0D0)) , INTENT(OUT) , DIMENSION(nlmax,2,2) :: dzdsgpz
   COMPLEX(kind(1.0D0)) , INTENT(OUT) , DIMENSION(nlmax,2,2) :: dzdstr
   COMPLEX(kind(1.0D0)) , INTENT(OUT) , DIMENSION(nlmax,2,2) :: dzddip
   COMPLEX(kind(1.0D0)) , INTENT(OUT) , DIMENSION(nlmax,2,2) :: dzdsla
!
! Local variable declarations rewritten by SPAG
!
   REAL(kind(1.0D0)) :: a12dif , asg , axx , axy , axydif , axysum , ayy , bsg , c2bs , c2dip , c2sla , c2str , cdip , csg , csla ,&
                      & cstr , daxxddip , daxxdsgpx , daxxdsgpy , daxxdsgpz , daxxdsla , daxxdstr , daxyddip , daxydsgpx ,         &
                      & daxydsgpy , daxydsgpz , daxydsla , daxydstr , dayyddip , dayydsgpx , dayydsgpy , dayydsgpz , dayydsla ,    &
                      & dayydstr , dbsgpz , dda1 , dda2 , ddbs , dsg , dsgpxy , dsigxxddip , dsigxxdsgpx , dsigxxdsgpy ,           &
                      & dsigxxdsgpz , dsigxxdsla , dsigxxdstr , dsigxyddip , dsigxydsgpx , dsigxydsgpy , dsigxydsgpz , dsigxydsla ,&
                      & dsigxydstr , dsigxzddip , dsigxzdsgpx , dsigxzdsgpy , dsigxzdsgpz , dsigxzdsla , dsigxzdstr , dsigyyddip , &
                      & dsigyydsgpx , dsigyydsgpy , dsigyydsgpz , dsigyydsla , dsigyydstr , dsigyzddip , dsigyzdsgpx ,             &
                      & dsigyzdsgpy , dsigyzdsgpz , dsigyzdsla , dsigyzdstr , dsigzzddip , dsigzzdsgpx , dsigzzdsgpy ,             &
                      & dsigzzdsgpz , dsigzzdsla , dsigzzdstr , hd , p1 , p1a , p1b
   INTEGER :: ix , iy , layer
   REAL(kind(1.0D0)) :: p2 , rdip , rsla , rstr , s2bs , s2dip , s2sla , s2str , sdip , sgpx , sgpy , sgpz , sigxx , sigxy ,       &
                      & sigxz , sigyy , sigyz , sigzz , ssla , sstr
!
! End of declarations rewritten by SPAG
!
!
! PARAMETER definitions rewritten by SPAG
!
!
! Dummy argument declarations rewritten by SPAG
!
!
! Local variable declarations rewritten by SPAG
!
!
! End of declarations rewritten by SPAG
!
!     ================================================================
!     Computes, from the sensitivities of the impedance with respect
!     to the effective azimuthal anisotropy parameters, sensitivities
!     with respect to the true physical anisotropy parameters, i.e.,
!     with respect to the principal conductivities and elementary
!     anisotropy directions, strike, dip, and slant.
!
!
!
!
!
   DO layer = 1 , nl
      hd = 1.D3*dble(h(layer))
      sgpx = 1.D0/dble(rop(layer,1))
      sgpy = 1.D0/dble(rop(layer,2))
      sgpz = 1.D0/dble(rop(layer,3))
!
      rstr = pi*dble(ustr(layer))/180.D0
      rdip = pi*dble(udip(layer))/180.D0
      rsla = pi*dble(usla(layer))/180.D0
!
      cstr = dcos(rstr)
      sstr = dsin(rstr)
      cdip = dcos(rdip)
      sdip = dsin(rdip)
      csla = dcos(rsla)
      ssla = dsin(rsla)
      c2str = dcos(2.D0*rstr)
      s2str = dsin(2.D0*rstr)
      c2dip = dcos(2.D0*rdip)
      s2dip = dsin(2.D0*rdip)
      c2sla = dcos(2.D0*rsla)
      s2sla = dsin(2.D0*rsla)
!
      dsgpxy = sgpx - sgpy
      asg = sgpx*csla**2.D0 + sgpy*ssla**2.D0
      bsg = sgpx*ssla**2.D0 + sgpy*csla**2.D0
      csg = bsg*cdip**2.D0 + sgpz*sdip**2.D0
      dsg = bsg*sdip**2.D0 + sgpz*cdip**2.D0
      dbsgpz = bsg - sgpz
!
!> sigxx and its derivatives
!
      sigxx = asg*cstr**2.D0 + csg*sstr**2.D0 - 0.5D0*dsgpxy*cdip*s2str*s2sla
!
      dsigxxdsgpx = (csla*cstr-ssla*cdip*sstr)**2.D0
      dsigxxdsgpy = (ssla*cstr+csla*cdip*sstr)**2.D0
      dsigxxdsgpz = (sdip*sstr)**2.D0
      dsigxxdstr = (-asg+csg)*s2str - dsgpxy*cdip*s2sla*c2str
      dsigxxddip = -dbsgpz*s2dip*sstr**2.D0 + 0.5D0*dsgpxy*sdip*s2sla*s2str
      dsigxxdsla = dsgpxy*((sstr*cdip)**2.D0*s2sla-cstr**2.D0*s2sla-cdip*s2str*c2sla)
!
!> sigxy and its derivatives
!
      sigxy = 0.5D0*(asg-csg)*s2str + 0.5D0*dsgpxy*cdip*c2str*s2sla
!
      dsigxydsgpx = 0.5D0*((csla**2.D0-(ssla*cdip)**2.D0)*s2str+cdip*c2str*s2sla)
      dsigxydsgpy = 0.5D0*((ssla**2.D0-(csla*cdip)**2.D0)*s2str-cdip*c2str*s2sla)
      dsigxydsgpz = -0.5D0*sdip**2.D0*s2str
      dsigxydstr = (asg-csg)*c2str - dsgpxy*cdip*s2str*s2sla
      dsigxyddip = 0.5D0*dbsgpz*s2dip*s2str - 0.5D0*dsgpxy*sdip*c2str*s2sla
      dsigxydsla = 0.5D0*dsgpxy*(2.D0*cdip*c2str*c2sla-(1.D0+cdip**2.D0)*s2str*s2sla)
!
!> sigxz and its derivatives
!
      sigxz = 0.5D0*dsgpxy*sdip*cstr*s2sla - 0.5D0*dbsgpz*s2dip*sstr
!
      dsigxzdsgpx = (cstr*csla-sstr*cdip*ssla)*ssla*sdip
      dsigxzdsgpy = -(cstr*ssla+sstr*cdip*csla)*csla*sdip
      dsigxzdsgpz = 0.5D0*s2dip*sstr
      dsigxzdstr = -0.5D0*dsgpxy*sdip*sstr*s2sla - 0.5D0*dbsgpz*s2dip*cstr
      dsigxzddip = 0.5D0*dsgpxy*cdip*cstr*s2sla - dbsgpz*c2dip*sstr
      dsigxzdsla = 0.5D0*dsgpxy*(2.D0*sdip*cstr*c2sla-s2dip*sstr*s2sla)
!
!> sigyy and its derivatives
!
      sigyy = asg*sstr**2.D0 + csg*cstr**2.D0 + 0.5D0*dsgpxy*cdip*s2str*s2sla
!
      dsigyydsgpx = (sstr*csla+cdip*cstr*ssla)**2.D0
      dsigyydsgpy = (sstr*ssla-cdip*cstr*csla)**2.D0
      dsigyydsgpz = (cstr*sdip)**2.D0
      dsigyydstr = (asg-csg)*s2str + dsgpxy*cdip*c2str*s2sla
      dsigyyddip = -dbsgpz*s2dip*cstr**2.D0 - 0.5D0*dsgpxy*sdip*s2str*s2sla
      dsigyydsla = dsgpxy*(cdip*s2str*c2sla+(cdip*cstr)**2.D0*s2sla-sstr**2.D0*s2sla)
!
!> sigyz and its derivatives
!
      sigyz = 0.5D0*dsgpxy*sdip*sstr*s2sla + 0.5D0*dbsgpz*s2dip*cstr
!
      dsigyzdsgpx = (sstr*csla+cdip*cstr*ssla)*sdip*ssla
      dsigyzdsgpy = -(sstr*ssla-cdip*cstr*csla)*sdip*csla
      dsigyzdsgpz = -0.5D0*s2dip*cstr
      dsigyzdstr = 0.5D0*dsgpxy*sdip*cstr*s2sla - 0.5D0*dbsgpz*s2dip*sstr
      dsigyzddip = 0.5D0*dsgpxy*cdip*sstr*s2sla + dbsgpz*c2dip*cstr
      dsigyzdsla = 0.5D0*dsgpxy*(2.D0*sdip*sstr*c2sla+s2dip*cstr*s2sla)
!
!> sigzz and its derivatives
!
      sigzz = dsg
!
      dsigzzdsgpx = (sdip*ssla)**2.D0
      dsigzzdsgpy = (sdip*csla)**2.D0
      dsigzzdsgpz = cdip**2.D0
      dsigzzdstr = 0.D0
      dsigzzddip = dbsgpz*s2dip
      dsigzzdsla = dsgpxy*s2sla*sdip**2.D0
!
!> axx and its derivatives
!
      axx = sigxx - sigxz*sigxz/sigzz
!
      p1 = 2.D0*sigxz/sigzz
      p2 = (sigxz/sigzz)**2.D0
      daxxdsgpx = dsigxxdsgpx - p1*dsigxzdsgpx + p2*dsigzzdsgpx
      daxxdsgpy = dsigxxdsgpy - p1*dsigxzdsgpy + p2*dsigzzdsgpy
      daxxdsgpz = dsigxxdsgpz - p1*dsigxzdsgpz + p2*dsigzzdsgpz
      daxxdstr = dsigxxdstr - p1*dsigxzdstr + p2*dsigzzdstr
      daxxddip = dsigxxddip - p1*dsigxzddip + p2*dsigzzddip
      daxxdsla = dsigxxdsla - p1*dsigxzdsla + p2*dsigzzdsla
!
!> ayy and its derivatives
!
      ayy = sigyy - sigyz*sigyz/sigzz
!
      p1 = 2.D0*sigyz/sigzz
      p2 = (sigyz/sigzz)**2.D0
      dayydsgpx = dsigyydsgpx - p1*dsigyzdsgpx + p2*dsigzzdsgpx
      dayydsgpy = dsigyydsgpy - p1*dsigyzdsgpy + p2*dsigzzdsgpy
      dayydsgpz = dsigyydsgpz - p1*dsigyzdsgpz + p2*dsigzzdsgpz
      dayydstr = dsigyydstr - p1*dsigyzdstr + p2*dsigzzdstr
      dayyddip = dsigyyddip - p1*dsigyzddip + p2*dsigzzddip
      dayydsla = dsigyydsla - p1*dsigyzdsla + p2*dsigzzdsla
!
!> axy and its derivatives
!
      axy = sigxy - sigxz*sigyz/sigzz
!
      p1a = sigxz/sigzz
      p1b = sigyz/sigzz
      p2 = p1a*p1b
      daxydsgpx = dsigxydsgpx - p1a*dsigyzdsgpx - p1b*dsigxzdsgpx + p2*dsigzzdsgpx
      daxydsgpy = dsigxydsgpy - p1a*dsigyzdsgpy - p1b*dsigxzdsgpy + p2*dsigzzdsgpy
      daxydsgpz = dsigxydsgpz - p1a*dsigyzdsgpz - p1b*dsigxzdsgpz + p2*dsigzzdsgpz
      daxydstr = dsigxydstr - p1a*dsigyzdstr - p1b*dsigxzdstr + p2*dsigzzdstr
      daxyddip = dsigxyddip - p1a*dsigyzddip - p1b*dsigxzddip + p2*dsigzzddip
      daxydsla = dsigxydsla - p1a*dsigyzdsla - p1b*dsigxzdsla + p2*dsigzzdsla
!
      c2bs = dcos(2.D0*blt(layer))
      s2bs = dsin(2.D0*blt(layer))
      a12dif = al(layer) - at(layer)
!
!> dZ/d(sgpx)
!
      axysum = 0.5D0*(daxxdsgpx+dayydsgpx)
      axydif = 0.5D0*(daxxdsgpx-dayydsgpx)
      dda1 = axysum + axydif*c2bs + daxydsgpx*s2bs
      dda2 = axysum - axydif*c2bs - daxydsgpx*s2bs
!
!> Intrinsic function TINY(real_argument) (Microsoft/Compaq Fortran)
!> gives the smallest number in the model representing the same type
!> and kind parameters as the argument. Alternatives are IMSL functions
!> AMACH(1) or DMACH(1) that give the smallest normalized positive
!> number in the computer's single-precision or double-precision
!> arithmetic, respectively, or adequate functions for machine constants
!> determination on other systems
!
      IF ( dabs(a12dif)<tiny(a12dif) ) THEN
         ddbs = 0.D0
      ELSE
         ddbs = (daxydsgpx*c2bs-axydif*s2bs)/a12dif
      ENDIF
      DO ix = 1 , 2
         DO iy = 1 , 2
            dzdsgpx(layer,ix,iy) = dzdal(layer,ix,iy)*dda1 + dzdat(layer,ix,iy)*dda2 + dzdbs(layer,ix,iy)*ddbs
         ENDDO
      ENDDO
!
!> dZ/d(sgpy)
!
      axysum = 0.5D0*(daxxdsgpy+dayydsgpy)
      axydif = 0.5D0*(daxxdsgpy-dayydsgpy)
      dda1 = axysum + axydif*c2bs + daxydsgpy*s2bs
      dda2 = axysum - axydif*c2bs - daxydsgpy*s2bs
      IF ( dabs(a12dif)<tiny(a12dif) ) THEN
         ddbs = 0.D0
      ELSE
         ddbs = (daxydsgpy*c2bs-axydif*s2bs)/a12dif
      ENDIF
      DO ix = 1 , 2
         DO iy = 1 , 2
            dzdsgpy(layer,ix,iy) = dzdal(layer,ix,iy)*dda1 + dzdat(layer,ix,iy)*dda2 + dzdbs(layer,ix,iy)*ddbs
         ENDDO
      ENDDO
!
!> dZ/d(sgpz)
!
      axysum = 0.5D0*(daxxdsgpz+dayydsgpz)
      axydif = 0.5D0*(daxxdsgpz-dayydsgpz)
      dda1 = axysum + axydif*c2bs + daxydsgpz*s2bs
      dda2 = axysum - axydif*c2bs - daxydsgpz*s2bs
      IF ( dabs(a12dif)<tiny(a12dif) ) THEN
         ddbs = 0.D0
      ELSE
         ddbs = (daxydsgpz*c2bs-axydif*s2bs)/a12dif
      ENDIF
      DO ix = 1 , 2
         DO iy = 1 , 2
            dzdsgpz(layer,ix,iy) = dzdal(layer,ix,iy)*dda1 + dzdat(layer,ix,iy)*dda2 + dzdbs(layer,ix,iy)*ddbs
         ENDDO
      ENDDO
!
!> dZ/d(strike)
!
      axysum = 0.5D0*(daxxdstr+dayydstr)
      axydif = 0.5D0*(daxxdstr-dayydstr)
      dda1 = axysum + axydif*c2bs + daxydstr*s2bs
      dda2 = axysum - axydif*c2bs - daxydstr*s2bs
      IF ( dabs(a12dif)<tiny(a12dif) ) THEN
         ddbs = 0.D0
      ELSE
         ddbs = (daxydstr*c2bs-axydif*s2bs)/a12dif
      ENDIF
      DO ix = 1 , 2
         DO iy = 1 , 2
            dzdstr(layer,ix,iy) = dzdal(layer,ix,iy)*dda1 + dzdat(layer,ix,iy)*dda2 + dzdbs(layer,ix,iy)*ddbs
         ENDDO
      ENDDO
!
!> dZ/d(dip)
!
      axysum = 0.5D0*(daxxddip+dayyddip)
      axydif = 0.5D0*(daxxddip-dayyddip)
      dda1 = axysum + axydif*c2bs + daxyddip*s2bs
      dda2 = axysum - axydif*c2bs - daxyddip*s2bs
      IF ( dabs(a12dif)<tiny(a12dif) ) THEN
         ddbs = 0.D0
      ELSE
         ddbs = (daxyddip*c2bs-axydif*s2bs)/a12dif
      ENDIF
      DO ix = 1 , 2
         DO iy = 1 , 2
            dzddip(layer,ix,iy) = dzdal(layer,ix,iy)*dda1 + dzdat(layer,ix,iy)*dda2 + dzdbs(layer,ix,iy)*ddbs
         ENDDO
      ENDDO
!
!> dZ/d(slant)
!
      axysum = 0.5D0*(daxxdsla+dayydsla)
      axydif = 0.5D0*(daxxdsla-dayydsla)
      dda1 = axysum + axydif*c2bs + daxydsla*s2bs
      dda2 = axysum - axydif*c2bs - daxydsla*s2bs
      IF ( dabs(a12dif)<tiny(a12dif) ) THEN
         ddbs = 0.D0
      ELSE
         ddbs = (daxydsla*c2bs-axydif*s2bs)/a12dif
      ENDIF
      DO ix = 1 , 2
         DO iy = 1 , 2
            dzdsla(layer,ix,iy) = dzdal(layer,ix,iy)*dda1 + dzdat(layer,ix,iy)*dda2 + dzdbs(layer,ix,iy)*ddbs
         ENDDO
      ENDDO
!
   ENDDO
!
END SUBROUTINE zs1anis
!*==rotz.f90 processed by SPAG 8.04RA 21:21  7 Nov 2025
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!===> END SUBROUTINE ZS1ANIS
!
!
!===> SUBROUTINE ROTZ
!     =============================
SUBROUTINE rotz(za,betrad,zb)
   USE iso_fortran_env
   USE ISO_FORTRAN_ENV                 
   IMPLICIT NONE
!
! Dummy argument declarations rewritten by SPAG
!
   COMPLEX(kind(1.0D0)) , INTENT(IN) , DIMENSION(2,2) :: za
   REAL(kind(1.0D0)) , INTENT(IN) :: betrad
   COMPLEX(kind(1.0D0)) , INTENT(OUT) , DIMENSION(2,2) :: zb
!
! Local variable declarations rewritten by SPAG
!
   REAL(kind(1.0D0)) :: co2 , si2
   COMPLEX(kind(1.0D0)) :: dif1 , dif2 , sum1 , sum2
!
! End of declarations rewritten by SPAG
!
!
! Dummy argument declarations rewritten by SPAG
!
!
! Local variable declarations rewritten by SPAG
!
!
! End of declarations rewritten by SPAG
!
!     =============================
!     Rotates the impedance ZA by BETRAD (in radians) to obtain ZB
!
!
!
   co2 = dcos(2.D0*betrad)
   si2 = dsin(2.D0*betrad)
!
   sum1 = za(1,1) + za(2,2)
   sum2 = za(1,2) + za(2,1)
   dif1 = za(1,1) - za(2,2)
   dif2 = za(1,2) - za(2,1)
!
   zb(1,1) = 0.5D0*(sum1+dif1*co2+sum2*si2)
   zb(1,2) = 0.5D0*(dif2+sum2*co2-dif1*si2)
   zb(2,1) = 0.5D0*(-dif2+sum2*co2-dif1*si2)
   zb(2,2) = 0.5D0*(sum1-dif1*co2-sum2*si2)
!
END SUBROUTINE rotz
!*==rotzs.f90 processed by SPAG 8.04RA 21:21  7 Nov 2025
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!===> END SUBROUTINE ROTZ
!
!
!===> SUBROUTINE ROTZS
!     =======================================
SUBROUTINE rotzs(dza,nla,la,betrad,dzb)
   USE iso_fortran_env
   USE ISO_FORTRAN_ENV                 
   IMPLICIT NONE
!
! PARAMETER definitions rewritten by SPAG
!
   INTEGER , PARAMETER :: nlmax = 1001
!
! Dummy argument declarations rewritten by SPAG
!
   COMPLEX(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax,2,2) :: dza
   INTEGER , INTENT(IN) :: nla
   INTEGER , INTENT(IN) :: la
   REAL(kind(1.0D0)) , INTENT(IN) :: betrad
   COMPLEX(kind(1.0D0)) , INTENT(OUT) , DIMENSION(nlmax,2,2) :: dzb
!
! Local variable declarations rewritten by SPAG
!
   REAL(kind(1.0D0)) :: co2 , si2
   COMPLEX(kind(1.0D0)) :: dif1 , dif2 , sum1 , sum2
   INTEGER :: l
!
! End of declarations rewritten by SPAG
!
!
! PARAMETER definitions rewritten by SPAG
!
!
! Dummy argument declarations rewritten by SPAG
!
!
! Local variable declarations rewritten by SPAG
!
!
! End of declarations rewritten by SPAG
!
!     =======================================
!     Rotates the sensitivities DZA of layers from LA to NLA by
!     BETRAD (in radians) to obtain DZB
!
!
!
   co2 = dcos(2.D0*betrad)
   si2 = dsin(2.D0*betrad)
!
   DO l = la , nla
      sum1 = dza(l,1,1) + dza(l,2,2)
      sum2 = dza(l,1,2) + dza(l,2,1)
      dif1 = dza(l,1,1) - dza(l,2,2)
      dif2 = dza(l,1,2) - dza(l,2,1)
!
      dzb(l,1,1) = 0.5D0*(sum1+dif1*co2+sum2*si2)
      dzb(l,1,2) = 0.5D0*(dif2+sum2*co2-dif1*si2)
      dzb(l,2,1) = 0.5D0*(-dif2+sum2*co2-dif1*si2)
      dzb(l,2,2) = 0.5D0*(sum1-dif1*co2-sum2*si2)
   ENDDO
!
END SUBROUTINE rotzs
!*==prep_aniso.f90 processed by SPAG 8.04RA 21:21  7 Nov 2025
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!===> END SUBROUTINE ROTZS
!
!
!===> SUBROUTINE CPANIS
!     =====================================================
SUBROUTINE prep_aniso(rop,ustr,udip,usla,nl,sg,al,at,blt)
   USE iso_fortran_env
   USE ISO_FORTRAN_ENV                 
   IMPLICIT NONE
!
! PARAMETER definitions rewritten by SPAG
!
   REAL(kind(1.0D0)) , PARAMETER :: pi = 3.14159265358979323846264338327950288D0
   INTEGER , PARAMETER :: nlmax = 1001
!
! Dummy argument declarations rewritten by SPAG
!
   REAL(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax,3) :: rop
   REAL(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax) :: ustr
   REAL(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax) :: udip
   REAL(kind(1.0D0)) , INTENT(IN) , DIMENSION(nlmax) :: usla
   INTEGER , INTENT(IN) :: nl
   REAL(kind(1.0D0)) , INTENT(INOUT) , DIMENSION(nlmax,3,3) :: sg
   REAL(kind(1.0D0)) , INTENT(OUT) , DIMENSION(nlmax) :: al
   REAL(kind(1.0D0)) , INTENT(OUT) , DIMENSION(nlmax) :: at
   REAL(kind(1.0D0)) , INTENT(INOUT) , DIMENSION(nlmax) :: blt
!
! Local variable declarations rewritten by SPAG
!
   REAL(kind(1.0D0)) :: axx , axy , ayx , ayy , c2ps , c2th , cfi , cps , csps , csth , cth , da12 , pom1 , pom2 , pom3 , rdip ,   &
                      & rsla , rstr , s2ps , s2th , sfi , sps , sth
   INTEGER :: j , layer
   REAL(kind(1.0D0)) , DIMENSION(nlmax,3) :: sgp
!
! End of declarations rewritten by SPAG
!
!
! PARAMETER definitions rewritten by SPAG
!
!
! Dummy argument declarations rewritten by SPAG
!
!
! Local variable declarations rewritten by SPAG
!
!
! End of declarations rewritten by SPAG
!
!     =====================================================
!     Computes effective azimuthal anisotropy parameters in
!     anisotropic layers from their principal resistivities
!     and elementary anisotropic directions
!
!     Input:
!     ------
!     ROP(NLMAX,3).....real*8 array of principal resistivities
!                      of the anisotropic layers in Ohm*m
!     USTR(NLMAX)......real*8 array with the anisotropy strike
!                      (first Euler's rotation) of the layers
!                      in degrees
!     UDIP(NLMAX)......real*8 array with the anisotropy dip
!                      (second Euler's rotation) of the layers
!                      in degrees
!     USLA(NLMAX)......real*8 array with the anisotropy slant
!                      (third Euler's rotation) of the layers
!                      in degrees
!     NL...............integer*4 number of layers in the model
!                      including the basement, NL<=NLMAX
!
!     Output:
!     -------
!     SG(NLMAX,3,3)....real*8 array with conductivity tensor
!                      elements of the individual layers of the
!                      anisotropic model in S/m. The tensor is
!                      always symmetric and positive definite
!     AL(NLMAX)........real*8 array with the maximum effective
!                      horizontal conductivities of the layers
!                      in S/m
!     AT(NLMAX)........real*8 array with the minimum effective
!                      horizontal conductivities of the layers
!                      in S/m
!     BLT(NLMAX).......real*8 array with the effective horizontal
!                      anisotropy strike (direction of the maximum
!                      conductivity) of the layers in RADIANS
!
!
!
!
!
   DO layer = 1 , nl
!
      DO j = 1 , 3
         sgp(layer,j) = 1.D0/dble(rop(layer,j))
      ENDDO
!
      rstr = pi*dble(ustr(layer))/180.D0
      rdip = pi*dble(udip(layer))/180.D0
      rsla = pi*dble(usla(layer))/180.D0
      sps = dsin(rstr)
      cps = dcos(rstr)
      sth = dsin(rdip)
      cth = dcos(rdip)
      sfi = dsin(rsla)
      cfi = dcos(rsla)
      pom1 = sgp(layer,1)*cfi*cfi + sgp(layer,2)*sfi*sfi
      pom2 = sgp(layer,1)*sfi*sfi + sgp(layer,2)*cfi*cfi
      pom3 = (sgp(layer,1)-sgp(layer,2))*sfi*cfi
      c2ps = cps*cps
      s2ps = sps*sps
      c2th = cth*cth
      s2th = sth*sth
      csps = cps*sps
      csth = cth*sth
!
!> Conductivity tensor
!
      sg(layer,1,1) = pom1*c2ps + pom2*s2ps*c2th - 2.*pom3*cth*csps + sgp(layer,3)*s2th*s2ps
      sg(layer,1,2) = pom1*csps - pom2*c2th*csps + pom3*cth*(c2ps-s2ps) - sgp(layer,3)*s2th*csps
      sg(layer,1,3) = -pom2*csth*sps + pom3*sth*cps + sgp(layer,3)*csth*sps
      sg(layer,2,1) = sg(layer,1,2)
      sg(layer,2,2) = pom1*s2ps + pom2*c2ps*c2th + 2.*pom3*cth*csps + sgp(layer,3)*s2th*c2ps
      sg(layer,2,3) = pom2*csth*cps + pom3*sth*sps - sgp(layer,3)*csth*cps
      sg(layer,3,1) = sg(layer,1,3)
      sg(layer,3,2) = sg(layer,2,3)
      sg(layer,3,3) = pom2*s2th + sgp(layer,3)*c2th
!
!> Effective horizontal 2*2 conductivity tensor
!
      axx = sg(layer,1,1) - sg(layer,1,3)*sg(layer,3,1)/sg(layer,3,3)
      axy = sg(layer,1,2) - sg(layer,1,3)*sg(layer,3,2)/sg(layer,3,3)
      ayx = sg(layer,2,1) - sg(layer,3,1)*sg(layer,2,3)/sg(layer,3,3)
      ayy = sg(layer,2,2) - sg(layer,2,3)*sg(layer,3,2)/sg(layer,3,3)
!
!> Principal conductivities and anisotropy strike of the effective
!> horizontal conductivity tensor
!
      da12 = dsqrt((axx-ayy)*(axx-ayy)+4.D0*axy*ayx)
      al(layer) = 0.5D0*(axx+ayy+da12)
      at(layer) = 0.5D0*(axx+ayy-da12)
      IF ( da12>=tiny(da12) ) THEN
         blt(layer) = (axx-ayy)/da12
         blt(layer) = 0.5D0*dacos(blt(layer))
      ELSE
         blt(layer) = 0.D0
      ENDIF
      IF ( axy<0.D0 ) blt(layer) = -blt(layer)
!
   ENDDO
!
END SUBROUTINE prep_aniso
!*==zsprpg.f90 processed by SPAG 8.04RA 21:21  7 Nov 2025
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!===> END SUBROUTINE CPANIS
!
 
!
!===> SUBROUTINE ZSPRPG
!     ========================================================
SUBROUTINE zsprpg(dzbot,zbot,ztop,dz1,dz2,ag1,ag2,dztop)
   USE iso_fortran_env
   IMPLICIT NONE
!
! Dummy argument declarations rewritten by SPAG
!
   COMPLEX(kind(1.0D0)) , INTENT(IN) , DIMENSION(2,2) :: dzbot
   COMPLEX(kind(1.0D0)) , INTENT(IN) , DIMENSION(2,2) :: zbot
   COMPLEX(kind(1.0D0)) , INTENT(IN) , DIMENSION(2,2) :: ztop
   COMPLEX(kind(1.0D0)) , INTENT(IN) :: dz1
   COMPLEX(kind(1.0D0)) , INTENT(IN) :: dz2
   COMPLEX(kind(1.0D0)) :: ag1
   COMPLEX(kind(1.0D0)) :: ag2
   COMPLEX(kind(1.0D0)) , INTENT(OUT) , DIMENSION(2,2) :: dztop
!
! Local variable declarations rewritten by SPAG
!
   COMPLEX(kind(1.0D0)) :: ddtzbot , dn11 , dn12 , dn21 , dn22 , dtzbot , dzdenom , zdenom
   COMPLEX(kind(1.0D0)) , EXTERNAL :: dfm , dfp
!
! End of declarations rewritten by SPAG
!
!
! Dummy argument declarations rewritten by SPAG
!
!
! Local variable declarations rewritten by SPAG
!
!
! End of declarations rewritten by SPAG
!
!     ========================================================
!     Propagates the parametric sensitivity from the bottom (DZBOT)
!     to the top (DZTOP) of an anisotropic layer
!
!
!
!
!
   dtzbot = zbot(1,1)*zbot(2,2) - zbot(1,2)*zbot(2,1)
   zdenom = dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + zbot(1,2)*dfm(ag1)*dfp(ag2)/dz1 - zbot(2,1)*dfp(ag1)*dfm(ag2)/dz2 + dfp(ag1)      &
          & *dfp(ag2)
   ddtzbot = dzbot(1,1)*zbot(2,2) + zbot(1,1)*dzbot(2,2) - dzbot(1,2)*zbot(2,1) - zbot(1,2)*dzbot(2,1)
   dzdenom = ddtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + dzbot(1,2)*dfm(ag1)*dfp(ag2)/dz1 - dzbot(2,1)*dfp(ag1)*dfm(ag2)/dz2
!
   dn11 = 4.D0*dzbot(1,1)*cdexp(-ag1-ag2)
   dn12 = dzbot(1,2)*dfp(ag1)*dfp(ag2) - dzbot(2,1)*dfm(ag1)*dfm(ag2)*dz1/dz2 + ddtzbot*dfp(ag1)*dfm(ag2)/dz2
   dn21 = dzbot(2,1)*dfp(ag1)*dfp(ag2) - dzbot(1,2)*dfm(ag1)*dfm(ag2)*dz2/dz1 - ddtzbot*dfm(ag1)*dfp(ag2)/dz1
   dn22 = 4.D0*dzbot(2,2)*cdexp(-ag1-ag2)
!
   dztop(1,1) = (dn11-ztop(1,1)*dzdenom)/zdenom
   dztop(1,2) = (dn12-ztop(1,2)*dzdenom)/zdenom
   dztop(2,1) = (dn21-ztop(2,1)*dzdenom)/zdenom
   dztop(2,2) = (dn22-ztop(2,2)*dzdenom)/zdenom
!
END SUBROUTINE zsprpg
!*==zscua1.f90 processed by SPAG 8.04RA 21:21  7 Nov 2025
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!===> END SUBROUTINE ZSPRPG
!
!
!===> SUBROUTINE ZSCUA1
!     =====================================================
SUBROUTINE zscua1(zbot,ztop,a1,dz1,dz2,ag1,ag2,dztop)
   USE iso_fortran_env
   IMPLICIT NONE
!
! Dummy argument declarations rewritten by SPAG
!
   COMPLEX(kind(1.0D0)) , INTENT(IN) , DIMENSION(2,2) :: zbot
   COMPLEX(kind(1.0D0)) , INTENT(IN) , DIMENSION(2,2) :: ztop
   REAL(kind(1.0D0)) , INTENT(IN) :: a1
   COMPLEX(kind(1.0D0)) , INTENT(IN) :: dz1
   COMPLEX(kind(1.0D0)) , INTENT(IN) :: dz2
   COMPLEX(kind(1.0D0)) :: ag1
   COMPLEX(kind(1.0D0)) :: ag2
   COMPLEX(kind(1.0D0)) , INTENT(OUT) , DIMENSION(2,2) :: dztop
!
! Local variable declarations rewritten by SPAG
!
   COMPLEX(kind(1.0D0)) :: dapom , dbpom , dcpom , ddpom , depom , dfpom , dgpom , dn12 , dn21 , dtzbot , dzdenom , zdenom
   COMPLEX(kind(1.0D0)) , EXTERNAL :: dfm , dfp
!
! End of declarations rewritten by SPAG
!
!
! Dummy argument declarations rewritten by SPAG
!
!
! Local variable declarations rewritten by SPAG
!
!
! End of declarations rewritten by SPAG
!
!     =====================================================
!     Computes the derivative of the impedance tensor with respect
!     to the maximum horizontal conductivity (A1=AL) within an
!     anisotropic layer
!
!
!
!
   dtzbot = zbot(1,1)*zbot(2,2) - zbot(1,2)*zbot(2,1)
   zdenom = dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + zbot(1,2)*dfm(ag1)*dfp(ag2)/dz1 - zbot(2,1)*dfp(ag1)*dfm(ag2)/dz2 + dfp(ag1)      &
          & *dfp(ag2)
!
   dapom = dtzbot*dfm(ag2)/dz2 + zbot(1,2)*dfp(ag2)
   dbpom = dfp(ag2) - zbot(2,1)*dfm(ag2)/dz2
   dcpom = zbot(2,1)*dfp(ag2) - dz2*dfm(ag2)
   ddpom = dtzbot*dfp(ag2) + dz2*zbot(1,2)*dfm(ag2)
   depom = ag1*dfm(ag1)
   dfpom = (dfm(ag1)+ag1*dfp(ag1))/dz1
   dgpom = (dfm(ag1)-ag1*dfp(ag1))*dz1
!
   dzdenom = dfpom*dapom + depom*dbpom
   dn12 = depom*dapom - dgpom*dbpom
   dn21 = depom*dcpom - dfpom*ddpom
!
   dztop(1,1) = -0.5D0*ztop(1,1)*dzdenom/(a1*zdenom)
   dztop(1,2) = 0.5D0*(dn12-ztop(1,2)*dzdenom)/(a1*zdenom)
   dztop(2,1) = 0.5D0*(dn21-ztop(2,1)*dzdenom)/(a1*zdenom)
   dztop(2,2) = -0.5D0*ztop(2,2)*dzdenom/(a1*zdenom)
!
END SUBROUTINE zscua1
!*==zscua2.f90 processed by SPAG 8.04RA 21:21  7 Nov 2025
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!===> END SUBROUTINE ZSCUA1
!
!
!===> SUBROUTINE ZSCUA2
!     =====================================================
SUBROUTINE zscua2(zbot,ztop,a2,dz1,dz2,ag1,ag2,dztop)
   USE iso_fortran_env
   IMPLICIT NONE
!
! Dummy argument declarations rewritten by SPAG
!
   COMPLEX(kind(1.0D0)) , INTENT(IN) , DIMENSION(2,2) :: zbot
   COMPLEX(kind(1.0D0)) , INTENT(IN) , DIMENSION(2,2) :: ztop
   REAL(kind(1.0D0)) , INTENT(IN) :: a2
   COMPLEX(kind(1.0D0)) , INTENT(IN) :: dz1
   COMPLEX(kind(1.0D0)) , INTENT(IN) :: dz2
   COMPLEX(kind(1.0D0)) :: ag1
   COMPLEX(kind(1.0D0)) :: ag2
   COMPLEX(kind(1.0D0)) , INTENT(OUT) , DIMENSION(2,2) :: dztop
!
! Local variable declarations rewritten by SPAG
!
   COMPLEX(kind(1.0D0)) :: dapom , dbpom , dcpom , ddpom , depom , dfpom , dgpom , dn12 , dn21 , dtzbot , dzdenom , zdenom
   COMPLEX(kind(1.0D0)) , EXTERNAL :: dfm , dfp
!
! End of declarations rewritten by SPAG
!
!
! Dummy argument declarations rewritten by SPAG
!
!
! Local variable declarations rewritten by SPAG
!
!
! End of declarations rewritten by SPAG
!
!     =====================================================
!     Computes the derivative of the impedance tensor with respect
!     to the minimum horizontal conductivity (A2=AT) within an
!     anisotropic layer
!
!
!
!
   dtzbot = zbot(1,1)*zbot(2,2) - zbot(1,2)*zbot(2,1)
   zdenom = dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + zbot(1,2)*dfm(ag1)*dfp(ag2)/dz1 - zbot(2,1)*dfp(ag1)*dfm(ag2)/dz2 + dfp(ag1)      &
          & *dfp(ag2)
!
   dapom = dtzbot*dfm(ag1)/dz1 - zbot(2,1)*dfp(ag1)
   dbpom = dfp(ag1) + zbot(1,2)*dfm(ag1)/dz1
   dcpom = zbot(1,2)*dfp(ag1) + dz1*dfm(ag1)
   ddpom = dtzbot*dfp(ag1) - dz1*zbot(2,1)*dfm(ag1)
   depom = ag2*dfm(ag2)
   dfpom = (dfm(ag2)+ag2*dfp(ag2))/dz2
   dgpom = (dfm(ag2)-ag2*dfp(ag2))*dz2
!
   dzdenom = dfpom*dapom + depom*dbpom
   dn12 = depom*dcpom + dfpom*ddpom
   dn21 = -depom*dapom + dgpom*dbpom
!
   dztop(1,1) = -0.5D0*ztop(1,1)*dzdenom/(a2*zdenom)
   dztop(1,2) = 0.5D0*(dn12-ztop(1,2)*dzdenom)/(a2*zdenom)
   dztop(2,1) = 0.5D0*(dn21-ztop(2,1)*dzdenom)/(a2*zdenom)
   dztop(2,2) = -0.5D0*ztop(2,2)*dzdenom/(a2*zdenom)
!
END SUBROUTINE zscua2
!*==zscubs.f90 processed by SPAG 8.04RA 21:21  7 Nov 2025
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!===> END SUBROUTINE ZSCUA2
!
!
!===> SUBROUTINE ZSCUBS
!     ==================================================
SUBROUTINE zscubs(zbot,ztop,dz1,dz2,ag1,ag2,dztop)
   USE iso_fortran_env
   IMPLICIT NONE
!
! Dummy argument declarations rewritten by SPAG
!
   COMPLEX(kind(1.0D0)) , INTENT(IN) , DIMENSION(2,2) :: zbot
   COMPLEX(kind(1.0D0)) , INTENT(IN) , DIMENSION(2,2) :: ztop
   COMPLEX(kind(1.0D0)) , INTENT(IN) :: dz1
   COMPLEX(kind(1.0D0)) , INTENT(IN) :: dz2
   COMPLEX(kind(1.0D0)) :: ag1
   COMPLEX(kind(1.0D0)) :: ag2
   COMPLEX(kind(1.0D0)) , INTENT(INOUT) , DIMENSION(2,2) :: dztop
!
! Local variable declarations rewritten by SPAG
!
   COMPLEX(kind(1.0D0)) , EXTERNAL :: dfm , dfp
   COMPLEX(kind(1.0D0)) :: dpom , dtzbot , zdenom
   COMPLEX(kind(1.0D0)) , DIMENSION(2,2) :: dzbot
!
! End of declarations rewritten by SPAG
!
!
! Dummy argument declarations rewritten by SPAG
!
!
! Local variable declarations rewritten by SPAG
!
!
! End of declarations rewritten by SPAG
!
!     ==================================================
!     Computes the derivative of the impedance tensor with respect
!     to the effective horizontal anisotropy strike of the current
!     layer in the coordinate system aligned with that strike
!     direction
!
!
!
!
   dztop(1,1) = -ztop(1,2) - ztop(2,1)
   dztop(1,2) = ztop(1,1) - ztop(2,2)
   dztop(2,1) = dztop(1,2)
   dztop(2,2) = -dztop(1,1)
!
   dtzbot = zbot(1,1)*zbot(2,2) - zbot(1,2)*zbot(2,1)
   zdenom = dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + zbot(1,2)*dfm(ag1)*dfp(ag2)/dz1 - zbot(2,1)*dfp(ag1)*dfm(ag2)/dz2 + dfp(ag1)      &
          & *dfp(ag2)
!
   dzbot(1,1) = 4.D0*(zbot(1,2)+zbot(2,1))*cdexp(-ag1-ag2)
   dzbot(1,2) = (zbot(1,1)-zbot(2,2))*(dfm(ag1)*dfm(ag2)*dz1/dz2-dfp(ag1)*dfp(ag2))
   dzbot(2,1) = (zbot(1,1)-zbot(2,2))*(dfm(ag1)*dfm(ag2)*dz2/dz1-dfp(ag1)*dfp(ag2))
   dzbot(2,2) = -4.D0*(zbot(1,2)+zbot(2,1))*cdexp(-ag1-ag2)
!
   dpom = (zbot(1,1)-zbot(2,2))*(dfm(ag1)*dfp(ag2)/dz1-dfp(ag1)*dfm(ag2)/dz2)
!
   dztop(1,1) = dztop(1,1) + (dzbot(1,1)+dpom*ztop(1,1))/zdenom
   dztop(1,2) = dztop(1,2) + (dzbot(1,2)+dpom*ztop(1,2))/zdenom
   dztop(2,1) = dztop(2,1) + (dzbot(2,1)+dpom*ztop(2,1))/zdenom
   dztop(2,2) = dztop(2,2) + (dzbot(2,2)+dpom*ztop(2,2))/zdenom
!
END SUBROUTINE zscubs
!*==zscuh.f90 processed by SPAG 8.04RA 21:21  7 Nov 2025
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!===> END SUBROUTINE ZSCUBS
 
!===> SUBROUTINE ZSCUH
!     =======================================================
SUBROUTINE zscuh(zbot,ztop,a1,a2,dz1,dz2,ag1,ag2,dztop)
   USE iso_fortran_env
   IMPLICIT NONE
!
! Dummy argument declarations rewritten by SPAG
!
   COMPLEX(kind(1.0D0)) , INTENT(IN) , DIMENSION(2,2) :: zbot
   COMPLEX(kind(1.0D0)) , INTENT(IN) , DIMENSION(2,2) :: ztop
   REAL(kind(1.0D0)) , INTENT(IN) :: a1
   REAL(kind(1.0D0)) , INTENT(IN) :: a2
   COMPLEX(kind(1.0D0)) , INTENT(IN) :: dz1
   COMPLEX(kind(1.0D0)) , INTENT(IN) :: dz2
   COMPLEX(kind(1.0D0)) :: ag1
   COMPLEX(kind(1.0D0)) :: ag2
   COMPLEX(kind(1.0D0)) , INTENT(OUT) , DIMENSION(2,2) :: dztop
!
! Local variable declarations rewritten by SPAG
!
   COMPLEX(kind(1.0D0)) :: dapom1 , dapom2 , dbpom1 , dbpom2 , dcpom1 , dcpom2 , ddpom1 , ddpom2 , dn12 , dn21 , dtzbot , dzdenom ,&
                         & k1 , k2 , zdenom
   COMPLEX(kind(1.0D0)) , EXTERNAL :: dfm , dfp
!
! End of declarations rewritten by SPAG
!
!
! Dummy argument declarations rewritten by SPAG
!
!
! Local variable declarations rewritten by SPAG
!
!
! End of declarations rewritten by SPAG
!
!     =======================================================
!     Computes the derivative of the impedance tensor with respect
!     to the thickness (in m) of the current anisotropic layer
!
!
!
!
   dtzbot = zbot(1,1)*zbot(2,2) - zbot(1,2)*zbot(2,1)
   zdenom = dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + zbot(1,2)*dfm(ag1)*dfp(ag2)/dz1 - zbot(2,1)*dfp(ag1)*dfm(ag2)/dz2 + dfp(ag1)      &
          & *dfp(ag2)
!
   k1 = a1*dz1
   k2 = a2*dz2
!
   dapom1 = dtzbot*dfm(ag1)/dz1 - zbot(2,1)*dfp(ag1)
   dbpom1 = dfp(ag1) + zbot(1,2)*dfm(ag1)/dz1
   dcpom1 = zbot(1,2)*dfp(ag1) + dz1*dfm(ag1)
   ddpom1 = dtzbot*dfp(ag1) - dz1*zbot(2,1)*dfm(ag1)
!
   dapom2 = dtzbot*dfm(ag2)/dz2 + zbot(1,2)*dfp(ag2)
   dbpom2 = dfp(ag2) - zbot(2,1)*dfm(ag2)/dz2
   dcpom2 = zbot(2,1)*dfp(ag2) - dz2*dfm(ag2)
   ddpom2 = dtzbot*dfp(ag2) + dz2*zbot(1,2)*dfm(ag2)
!
   dzdenom = k1*(dapom2*dfp(ag1)/dz1+dbpom2*dfm(ag1)) + k2*(dapom1*dfp(ag2)/dz2+dbpom1*dfm(ag2))
   dn12 = k1*(dapom2*dfm(ag1)+dbpom2*dz1*dfp(ag1)) + k2*(dcpom1*dfm(ag2)+ddpom1*dfp(ag2)/dz2)
   dn21 = k1*(dcpom2*dfm(ag1)-ddpom2*dfp(ag1)/dz1) - k2*(dapom1*dfm(ag2)+dbpom1*dz2*dfp(ag2))
!
   dztop(1,1) = -ztop(1,1)*dzdenom/zdenom
   dztop(1,2) = (dn12-ztop(1,2)*dzdenom)/zdenom
   dztop(2,1) = (dn21-ztop(2,1)*dzdenom)/zdenom
   dztop(2,2) = -ztop(2,2)*dzdenom/zdenom
!
END SUBROUTINE zscuh
!*==dfm.f90 processed by SPAG 8.04RA 21:21  7 Nov 2025
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!===> END SUBROUTINE ZSCUH
 
!===> COMPLEX*16 FUNCTION DFM
!     ==========================
FUNCTION dfm(x)
   USE iso_fortran_env
   IMPLICIT NONE
!
! Function and Dummy argument declarations rewritten by SPAG
!
   COMPLEX(kind(1.0D0)) :: dfm
   COMPLEX(kind(1.0D0)) , INTENT(IN) :: x
!
! End of declarations rewritten by SPAG
!
!
! Function and Dummy argument declarations rewritten by SPAG
!
!
! End of declarations rewritten by SPAG
!
!     ==========================
!     Regularized hyperbolic sinus, dfm(x)=2.*sinh(x)/exp(x)
!
!
   dfm = 1.D0 - cdexp(-2.D0*x)
!
END FUNCTION dfm
!*==dfp.f90 processed by SPAG 8.04RA 21:21  7 Nov 2025
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!!SPAG Open source Personal, Educational or Academic User retired  NON-COMMERCIAL USE - Not for use on proprietary or closed source code
!===> END COMPLEX*16 FUNCTION DFM
!
!
!===> COMPLEX*16 FUNCTION DFP
!     ==========================
FUNCTION dfp(x)
   USE iso_fortran_env
   IMPLICIT NONE
!
! Function and Dummy argument declarations rewritten by SPAG
!
   COMPLEX(kind(1.0D0)) :: dfp
   COMPLEX(kind(1.0D0)) , INTENT(IN) :: x
!
! End of declarations rewritten by SPAG
!
!
! Function and Dummy argument declarations rewritten by SPAG
!
!
! End of declarations rewritten by SPAG
!
!     ==========================
!     Regularized hyperbolic cosinus, dfp(x)=2.*cosh(x)/exp(x)
!
!
   dfp = 1.D0 + cdexp(-2.D0*x)
!
END FUNCTION dfp
!===> END COMPLEX*16 FUNCTION DFM
 
