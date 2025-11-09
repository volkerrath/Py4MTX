c
c     ===================
      program zs1a_driver
c     ===================
c
c     Driver routine for the subroutine ZS1ANEF for the
c     computation of magnetotelluric impedances and their
c     parametric partial derivatives on the surface of a 1-D 
c     layered generally anisotropic medium
c
c     Algorithm description:
c     ------------------------------------------------------------------
c     Pek, J. and Santos, F. A. M., 2002. Magnetotelluric impedances and
c     parametric sensitivities for 1-D generally anisotropic layered 
c     media, Computers & Geosciences, in print.
c     ------------------------------------------------------------------
c
c     List of subroutines (13):
c     ZS1ANEF, ZS1ANIS, ROTZ, ROTZS, CPANIS, DPHASE, ZSPRPG, ZSCUA1,
c     ZSCUA2, ZSCUBS, ZSCUH, DFM, DFP
c
c     Compiled with Compaq Visual Fortran Professional Edition 6.5.0
c
      real*8 pi
      parameter(pi=3.14159265358979323846264338327950288d0)
      complex*16 ic
      parameter(ic=(0.d0,1.d0))
c
c> Maximum number of layers of the layered anisotropic model
c
      parameter(nlmax=1001)
c
      real*8 h(nlmax)
      real*8 rop(nlmax,3),ustr(nlmax),udip(nlmax),usla(nlmax)
      real*8 al(nlmax),at(nlmax),blt(nlmax),sg(nlmax,3,3)
      integer*4 layani(nlmax)
      complex*16 z(2,2)
      complex*16 dzdal(nlmax,2,2),dzdat(nlmax,2,2),dzdbs(nlmax,2,2)
      complex*16 dzdh(nlmax,2,2)
      real*8 omega,mu0,prev
      real*8 rapp(2,2), papp(2,2), per
      complex*16 dzdsgpx(nlmax,2,2),dzdsgpy(nlmax,2,2)
      complex*16 dzdsgpz(nlmax,2,2),dzdstr(nlmax,2,2)
      complex*16 dzddip(nlmax,2,2),dzdsla(nlmax,2,2)
c
c> Open data file with model parameters and read the model data
c
      open(1,file='an.dat')
c
c> NL .............. integer*4 number of layers including the basement
c> H(NLMAX) ........ real*8 array with layer thickesses, in km.
c>                   Thickness of the basement, H(NL), may be arbitrary
c> ROP(NLMAX,3) .... real*8 array of principal resistivities within the
c>                   layers, 3 for each layer, in Ohm*m
c> USTR(NLMAX) ..... real*8 array with anisotropy strikes of the layers,
c>                   in degrees
c> UDIP(NLMAX) ..... real*8 array with anisotropy dips of the layers,
c>                   in degrees
c> USLA(NLMAX) ..... real*8 array with anisotropy slants of the layers,
c>                   in degrees
c> LAYANI(NLMAX) ... integer*4 array of isotropy/anisotropy flags for
c>                   the individual layers. If a layer has all its
c>                   principal resistivities identical, it can be 
c>                   flagged as isotropic by setting LAYANI(I)=0
c

      read(1,*)nl
      do layer=1,nl
        read(1,*)h(layer),(rop(layer,i),i=1,3),
     &           ustr(layer),udip(layer),usla(layer),
     &           layani(layer)
      enddo
      close(1)
c
c> Permeability of the vacuum
c
      mu0=4.d-7*pi
c
c> Open files for results
c> ANS.RES contains model parameters, impedances, and parametric 
c>         sensitivities of the impedances
c> ANR.RES contains a simple table with the apparent resistivities
c>         and impedance phases for all periods involved
c
      open(2,file='ans.res')
      open(3,file='anr.dat')
c
c> Compute the equivalent parameters of an azimuthally anisotropic
c> medium for all layers in the stack
c> SG(NLMAX,3,3) ...... real*8 array with the components of the
c>                      conductivity tensor, in S/m, for the layers
c> AL(NLMAX) .......... real*8 array with the maximum horizontal
c>                      conductivities, in S/m, for the layers
c> AT(NLMAX) .......... real*8 array with the minimum horizontal
c>                      conductivities, in S/m, for the layers
c> BLT(NLMAX) ......... real*8 array with the effective horizontal
c>                      anisotropy strikes, in radians, for the
c>                      individual layers
c
      call cpanis(rop,ustr,udip,usla,nl,sg,al,at,blt)      
c
      write(2,'("=== Model parameters ====================")')
      write(2,*)
      do layer=1,nl
        write(2,'("+++> Layer = ",i5,", thickness in km = ",f12.4)')
     &                                                  layer,h(layer)
        write(2,'("--- Conductivity tensor, in S/m ---------")')
        write(2,'("s_xx  s_xy  s_xz",2x,3f14.5)')(sg(layer,1,j),j=1,3)
        write(2,'("s_yx  s_yy  s_yz",2x,3f14.5)')(sg(layer,2,j),j=1,3)
        write(2,'("s_zx  s_zy  s_zz",2x,3f14.5)')(sg(layer,3,j),j=1,3)
        write(2,'("--- CONDMAX,CONDMIN,ASTRIKE for eq.mod. --")')
        write(2,'(5x,3f14.5)')al(layer),at(layer),
     &                            180.d0*blt(layer)/pi
        write(2,*)
      enddo
      write(2,'("=========================================")')
c
c> Fixed period range from 10**(-3) to 10**(+5) seconds, the period step
c> is 0.2 in the logarithmic domain
c
      do iper=-30,50,2
c
c> Period of the MT field in seconds
c
        per=10.d0**(0.1d0*dble(iper))
c
c> Compute MT impedances and their parametric partial derivatives for 
c> 1-D layered anisotropic model. The sensitivities are computed for
c> the parameters of the effective horizontal anisotropy of the layers
c
c> Z(2,2) .............. complex*16 array with the elements of the 
c>                       impedance tensor on the surface of the medium,
c>                       in SI units (Ohm)
c> DZDAL(NLMAX,2,2) .... complex*16 array with partial derivatives of
c>                       the impedance tensor with respect to the 
c>                       maximum horizontal conductivities AL of the
c>                       layers, in SI units (Ohm**2*m)
c> DZDAT(NLMAX,2,2) .... complex*16 array with partial derivatives of
c>                       the impedance tensor with respect to the 
c>                       minimum horizontal conductivities AT of the
c>                       layers, in SI units (Ohm**2*m)
c> DZDBS(NLMAX,2,2) .... complex*16 array with partial derivatives of
c>                       the impedance tensor with respect to the
c>                       effective horizontal strikes BLT of the
c>                       layers, in SI units (Ohm/radian=Ohm)
c> DZDH(NLMAX,2,2) ..... complex*16 array with partial derivatives of
c>                       the impedance tensor with respect to the
c>                       thicknesses H of the layers, in Ohm/m
c
        call zs1anef(layani,h,al,at,blt,nl,per,z,dzdal,dzdat,dzdbs,dzdh)
c
c> Transform the parametric sensitivities with respect to the effective
c> horizontal anisotropy parameters into sensitivities with respect to
c> the true anisotropy parameters (principal conductivities and three
c> elementary anisotropy directions, strike, dip, and slant)
c
c> DZDSGPX(NLMAX,2,2) .. complex*16 array, derivatives of the impedance
c>                       tensor with respect to the first principal
c>                       conductivity of the layers, in Ohm**2*m
c> DZDSGPY(NLMAX,2,2) .. complex*16 array, derivatives of the impedance
c>                       tensor with respect to the second principal
c>                       conductivity of the layers, in Ohm**2*m
c> DZDSGPZ(NLMAX,2,2) .. complex*16 array, derivatives of the impedance
c>                       tensor with respect to the third principal
c>                       conductivity of the layers, in Ohm**2*m
c> DZDSTR(NLMAX,2,2) ... complex*16 array, derivatives of the impedance
c>                       tensor with respect to the anisotropy strike
c>                       of the layers, USTR, in Ohm/radian=Ohm
c> DZDDIP(NLMAX,2,2) ... complex*16 array, derivatives of the impedance
c>                       tensor with respect to the anisotropy dip
c>                       of the layers, UDIP, in Ohm/radian=Ohm
c> DZDSLA(NLMAX,2,2) ... complex*16 array, derivatives of the impedance
c>                       tensor with respect to the anisotropy slant
c>                       of the layers, USLA, in Ohm/radian=Ohm
c
        call zs1anis(layani,h,rop,ustr,udip,usla,al,at,blt,nl,
     &               per,z,dzdal,dzdat,dzdbs,dzdh,
     &               dzdsgpx,dzdsgpy,dzdsgpz,dzdstr,dzddip,dzdsla)
        write(2,*)
        write(2,'("=== Period and impedance tensor =========")')
        write(2,2001)
 2001 format(8x,"PERIOD",4x,"Re Zxx",6x,"Im Zxx",6x,"Re Zxy",6x,"Im Zxy"
     &)
        write(2,2002)
 2002 format(18x,"Re Zyx",6x,"Im Zyx",6x,"Re Zyy",6x,"Im Zyy")
        write(2,'(f14.5,4e12.4)')per,z(1,1),z(1,2)
        write(2,'(14x,4e12.4)')z(2,1),z(2,2)
c
        do il=1,nl
        write(2,'("--- Sensitivities, layer ",i5," ---------")')il
c
          if(layani(il).eq.0)then
            dzdal(il,1,1)=dzdal(il,1,1)+dzdat(il,1,1)
            dzdal(il,1,2)=dzdal(il,1,2)+dzdat(il,1,2)
            dzdal(il,2,1)=dzdal(il,2,1)+dzdat(il,2,1)
            dzdal(il,2,2)=dzdal(il,2,2)+dzdat(il,2,2)
c
            dzdat(il,1,1)=dzdal(il,1,1)
            dzdat(il,1,2)=dzdal(il,1,2)
            dzdat(il,2,1)=dzdal(il,2,1)
            dzdat(il,2,2)=dzdal(il,2,2)
          endif
c
          write(2,'(" CONDMAX ",i3,2x,4e12.4)')il,dzdal(il,1,1),
     &                                            dzdal(il,1,2)
          write(2,'(14x,4e12.4)')dzdal(il,2,1),dzdal(il,2,2)
          write(2,'(" CONDMIN ",i3,2x,4e12.4)')il,dzdat(il,1,1),
     &                                            dzdat(il,1,2)
          write(2,'(14x,4e12.4)')dzdat(il,2,1),dzdat(il,2,2)
          write(2,'(" ASTRIKE ",i3,2x,4e12.4)')il,dzdbs(il,1,1),
     &                                            dzdbs(il,1,2)
          write(2,'(14x,4e12.4)')dzdbs(il,2,1),dzdbs(il,2,2)
          write(2,'("  DEPTH  ",i3,2x,4e12.4)')il,dzdh(il,1,1),
     &                                            dzdh(il,1,2)
          write(2,'(14x,4e12.4)')dzdh(il,2,1),dzdh(il,2,2)
          write(2,'(".........................................")')
          write(2,'("  SGPX   ",i3,2x,4e12.4)')il,dzdsgpx(il,1,1),
     &                                            dzdsgpx(il,1,2)
          write(2,'(14x,4e12.4)')dzdsgpx(il,2,1),dzdsgpx(il,2,2)
          write(2,'("  SGPY   ",i3,2x,4e12.4)')il,dzdsgpy(il,1,1),
     &                                            dzdsgpy(il,1,2)
          write(2,'(14x,4e12.4)')dzdsgpy(il,2,1),dzdsgpy(il,2,2)
          write(2,'("  SGPZ   ",i3,2x,4e12.4)')il,dzdsgpz(il,1,1),
     &                                            dzdsgpz(il,1,2)
          write(2,'(14x,4e12.4)')dzdsgpz(il,2,1),dzdsgpz(il,2,2)
          write(2,'(" STRIKE  ",i3,2x,4e12.4)')il,dzdstr(il,1,1),
     &                                            dzdstr(il,1,2)
          write(2,'(14x,4e12.4)')dzdstr(il,2,1),dzdstr(il,2,2)
          write(2,'("   DIP   ",i3,2x,4e12.4)')il,dzddip(il,1,1),
     &                                            dzddip(il,1,2)
          write(2,'(14x,4e12.4)')dzddip(il,2,1),dzddip(il,2,2)
          write(2,'("  SLANT  ",i3,2x,4e12.4)')il,dzdsla(il,1,1),
     &                                            dzdsla(il,1,2)
          write(2,'(14x,4e12.4)')dzdsla(il,2,1),dzdsla(il,2,2)
        enddo
c
c> Compute apparent resistivities and impedance phases from the
c> impedance tensor
c
        omega=2.d0*pi/dble(per)
        prev=1.d0/(omega*mu0)
        do i=1,2
          do j=1,2
            rapp(i,j)=prev*cdabs(z(i,j))**2.d0
            papp(i,j)=180.d0*dphase(z(i,j))/pi
          enddo
        enddo
c
        write(2,'("--- Period, resistivities and phases ----")')
        write(2,2003)
 2003 format(8x,"PERIOD",4x,"RHOAxx",6x,"RHOAxy",6x,"RHOAyx",6x,"RHOAyy"
     &)
        write(2,2004)
 2004 format(18x,"PHIAxx",6x,"PHIAxy",6x,"PHIAyx",6x,"PHIAyy")
        write(2,'(f14.5,4(1pe12.4))')per,rapp(1,1),rapp(1,2),
     &                                   rapp(2,1),rapp(2,2)
        write(2,'(14x,4f12.2)')papp(1,1),papp(1,2),
     &                         papp(2,1),papp(2,2)
        write(3,'(f14.5,4(1pe12.4,2x,0pf12.2))')per,
     &                                          rapp(1,1),papp(1,1),
     &                                          rapp(1,2),papp(1,2),
     &                                          rapp(2,1),papp(2,1),
     &                                          rapp(2,2),papp(2,2)
      enddo
c
      close(2)
      close(3)
c
      stop
      end
c
c
c===> SUBROUTINE ZS1ANEF
c     =============================================
      subroutine zs1anef(layani,h,al,at,blt,nl,per,
     &                   z,dzdal,dzdat,dzdbs,dzdh)
c     =============================================
c     Stable impedance propagation procedure for 1-D layered
c     media with generally anisotropic layers. Computes both
c     the impedances and parametric sensitivities, the latter
c     only with respect to the effective horizontal anisotropy
c     parameters of the layer, i.e., longitudinal conductivity
c     AL (A1), transversal conductivity AT (A2), and the azi-
c     muthal anisotropy strike BLT (BS). For isotropic layers, 
c     the sensitivity with respect to the common conductivity
c     can be evaluated as well. Additionally, the sensitivity
c     with respect to the thickesses of the layers is computed.
c
c     Input:
c     ------
c     LAYANI(NLMAX)....integer*4 array of 0/1 indices that
c                      are set to 0 for layers to be treated
c                      consequently as isotropic. For layers
c                      with different princial conductivities
c                      the isotropy flag is changed automatically
c                      to 1 (i.e., anisotropic layer)
c     H(NLMAX).........real*8 array of layer thicknesses in km.
c                      Any value may be input as a thickness of                    
c                      the homogeneous basement
c     AL(NLMAX)........real*8 array of maximum horizontal con-
c                      ductivities of the layers in S/m
c     AT(NLMAX)........real*8 array of minimum horizontal con-
c                      ductivities of the layers in S/m
c     BLT(NLMAX).......real*8 array of equivalent horizontal
c                      anisotropy strikes of the layers in
c                      radians
c     NL...............integer*4 number of layers in the model
c                      including the basement, NL<=NLMAX
c     PER..............real*8 period of the elmg field in s
c
c     Output:
c     -------
c     Z(2,2)...........complex*16 array with the elements of the
c                      impedance tensor on the surface in SI units
c                      (Ohm). The arrangement is as follows
c                         | Z_xx  Z_xy | = | Z(1,1)  Z(1,2) |
c                         | Z_yx  Z_yy | = | Z(2,1)  Z(2,2) |
c     DZDAL(NLMAX,2,2).complex*16 array with partial derivatives of
c                      the impedance Z with respect to the maximum 
c                      horizontal conductivity of the layers, 
c                           DZDAL(I,J,K)=d{Z(J,K)}/d{AL(I)},
c                      in SI units (Ohm**2)
c     DZDAT(NLMAX,2,2).complex*16 array with partial derivatives of
c                      the impedance Z with respect to the minimum
c                      horizontal conductivity of the layers,
c                           DZDAT(I,J,K)=d{Z(J,K)}/d{AT(I)},
c                      in SI units (Ohm*2)
c     DZDBS(NLMAX,2,2).complex*16 array with partial derivatives of
c                      the impedance Z with respect to the effective
c                      horizontal anisotropy strike of the layers,
c                           DZDBS(I,J,K)=d{Z(J,K)}/d{BLT(I)},
c                      in SI units (Ohm/radian=Ohm)
c     DZDH(NLMAX,2,2)..complex*16 array with partial derivatives of
c                      the impedance Z with respect to the thickness
c                      of the layers
c                           DZDH(I,J,K)=d{Z(J,K)}/d{H(I)},
c                      in SI units (Ohm/m!!!), NOT in Ohm/km!!!
c
      real*8 pi
      parameter(pi=3.14159265358979323846264338327950288d0)
      complex*16 ic
      parameter(ic=(0.d0,1.d0))
c
      parameter(nlmax=1001)
c
      real*8 h(nlmax)
      real*8 al(nlmax),at(nlmax),blt(nlmax)
      real*8 per
      complex*16 z(2,2),zprd(2,2)
      complex*16 dzdal(nlmax,2,2),dzdat(nlmax,2,2),dzdbs(nlmax,2,2)
      complex*16 dzdh(nlmax,2,2)
      integer*4 layani(nlmax)
c
      complex*16 iom,k0,k1,k2
      complex*16 zrot(2,2),zbot(2,2),dtzbot,zdenom
      complex*16 dzdalrot(nlmax,2,2),dzdatrot(nlmax,2,2)
      complex*16 dzdbsrot(nlmax,2,2),dzdhrot(nlmax,2,2)
      complex*16 dztop(2,2),dzbot(2,2)
      complex*16 dz1,dz2,ag1,ag2
      real*8 omega,mu0
      real*8 a1,a2,bs,hd,bsref,a1is,a2is, c2bs, s2bs
c
      complex*16 dfm,dfp
c
      omega=2.d0*pi/dble(per)
      mu0=4.d-7*pi
      iom=-ic*omega*mu0
      k0=(1.d0-ic)*2.d-3*pi/dsqrt(10.d0*dble(per))
c
c> Compute the impedance on the top of the homogeneous
c> basement in the direction of its strike
c
      layer=nl
      a1=al(layer)
      a2=at(layer)
      bs=blt(layer)
c
      k1=k0*dsqrt(a1)
      k2=k0*dsqrt(a2)
      c2bs=dcos(2.d0*bs)
      s2bs=dsin(2.d0*bs)
      a1is=1.d0/dsqrt(a1)
      a2is=1.d0/dsqrt(a2)
c
      zrot(1,1)=0.d0
      zrot(1,2)=k0*a1is
      zrot(2,1)=-k0*a2is
      zrot(2,2)=0.d0
      call rotz(zrot,-bs,zprd)
c
c> In the isotropic case, compute the sensitivity with
c> respect to the isotropic basement conductivity
c
      if(layani(layer).eq.0.and.a1.eq.a2)then
        dzdalrot(layer,1,1)=0.d0
        dzdalrot(layer,1,2)=-0.5d0*zrot(1,2)/a1
        dzdalrot(layer,2,1)=0.d0
        dzdalrot(layer,2,2)=0.d0
c
        dzdatrot(layer,1,1)=0.d0
        dzdatrot(layer,1,2)=0.d0
        dzdatrot(layer,2,1)=-0.5d0*zrot(2,1)/a2
        dzdatrot(layer,2,2)=0.d0
c
        dzdbsrot(layer,1,1)=0.d0
        dzdbsrot(layer,1,2)=0.d0
        dzdbsrot(layer,2,1)=0.d0
        dzdbsrot(layer,2,2)=0.d0
c
c> For the anisotropic case, compute the sensitivies
c> separately with respect to the individual principal
c> conductivities and with respect to the effective
c> anisotropy strike of the basement
c
      else
        dzdalrot(layer,1,1)=0.d0
        dzdalrot(layer,1,2)=-0.5d0*zrot(1,2)/a1
        dzdalrot(layer,2,1)=0.d0
        dzdalrot(layer,2,2)=0.d0
c
        dzdatrot(layer,1,1)=0.d0
        dzdatrot(layer,1,2)=0.d0
        dzdatrot(layer,2,1)=-0.5d0*zrot(2,1)/a2
        dzdatrot(layer,2,2)=0.d0
c
        dzdbsrot(layer,1,1)=-zrot(1,2)-zrot(2,1)
        dzdbsrot(layer,1,2)=0.d0
        dzdbsrot(layer,2,1)=0.d0
        dzdbsrot(layer,2,2)=-dzdbsrot(layer,1,1)
        layani(layer)=1
      endif
c
c> Sensitivity of the basement impedance with respect to 
c> the thickness is always zero
c
      dzdhrot(layer,1,1)=0.d0
      dzdhrot(layer,1,2)=0.d0
      dzdhrot(layer,2,1)=0.d0
      dzdhrot(layer,2,2)=0.d0
c        
c> If no more layers are present in the model, rotate the
c> impedance and the sensitivities (except the one with respect
c> to the thickness) back into the original coordinate system 
c> and return
c
      if(nl.eq.1)then
        call rotz(zrot,-bs,z)
        call rotzs(dzdalrot,nl,layer,-bs,dzdal)
        call rotzs(dzdatrot,nl,layer,-bs,dzdat)
        call rotzs(dzdbsrot,nl,layer,-bs,dzdbs)
        dzdh(layer,1,1)=0.d0
        dzdh(layer,1,2)=0.d0
        dzdh(layer,2,1)=0.d0
        dzdh(layer,2,2)=0.d0
        return
      endif
c
c> Set the reference direction to the anisotropy strike of
c> the current layer
c
      bsref=bs
c
c> Process the rest of the layers above the basement
c
      nl1=nl-1
      do layer=nl1,1,-1
        hd=1.d+3*dble(h(layer))
        a1=al(layer)
        a2=at(layer)
        bs=blt(layer)
c
c> If the strike direction differs from that of the previous
c> layer, rotate the impedance and the parametric sensitivities
c> of all deeper layers into the coordinate system of the
c> current anisotropy strike. Store the rotated sensitivities 
c> in the positions of the original sensitivity variables
c
        layer1=layer+1
        dtzbot=zrot(1,1)*zrot(2,2)-zrot(1,2)*zrot(2,1)
        if(bs.ne.bsref.and.a1.ne.a2)then
          call rotz(zrot,bs-bsref,zbot)
          call rotzs(dzdalrot,nl,layer1,bs-bsref,dzdal)
          call rotzs(dzdatrot,nl,layer1,bs-bsref,dzdat)
          call rotzs(dzdbsrot,nl,layer1,bs-bsref,dzdbs)
          call rotzs(dzdhrot,nl,layer1,bs-bsref,dzdh)
c
c> If the anisotropy strike does not change, or for the isotropic
c> case, take the impedances from the top of the layer immediately
c> below as the bottom impedance for the current layer without any 
c> change. The same applies to the sensitivities of the deeper 
c> layers.
c
        else
          zbot(1,1)=zrot(1,1)
          zbot(1,2)=zrot(1,2)
          zbot(2,1)=zrot(2,1)
          zbot(2,2)=zrot(2,2)
          bs=bsref
          do ix=1,2
            do iy=1,2
              do il=layer1,nl
                dzdal(il,ix,iy)=dzdalrot(il,ix,iy)
                dzdat(il,ix,iy)=dzdatrot(il,ix,iy)
                dzdbs(il,ix,iy)=dzdbsrot(il,ix,iy)
                dzdh(il,ix,iy)=dzdhrot(il,ix,iy)
              enddo
            enddo
          enddo
        endif
c
        k1=k0*dsqrt(a1)
        k2=k0*dsqrt(a2)
        a1is=1.d0/dsqrt(a1)
        a2is=1.d0/dsqrt(a2)
        dz1=k0*a1is
        dz2=k0*a2is
        ag1=k1*hd
        ag2=k2*hd
c
c> Propagate the impedance tensor from the bottom to the top
c> of the current layer
c
        zdenom=dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2)+
     &         zbot(1,2)*dfm(ag1)*dfp(ag2)/dz1-
     &         zbot(2,1)*dfp(ag1)*dfm(ag2)/dz2+
     &         dfp(ag1)*dfp(ag2)
        zrot(1,1)=4.d0*zbot(1,1)*cdexp(-ag1-ag2)/zdenom
        zrot(1,2)=(zbot(1,2)*dfp(ag1)*dfp(ag2)-
     &             zbot(2,1)*dfm(ag1)*dfm(ag2)*dz1/dz2+
     &             dtzbot*dfp(ag1)*dfm(ag2)/dz2+
     &             dfm(ag1)*dfp(ag2)*dz1)/zdenom
        zrot(2,1)=(zbot(2,1)*dfp(ag1)*dfp(ag2)-
     &             zbot(1,2)*dfm(ag1)*dfm(ag2)*dz2/dz1-
     &             dtzbot*dfm(ag1)*dfp(ag2)/dz1-
     &             dfp(ag1)*dfm(ag2)*dz2)/zdenom
        zrot(2,2)=4.d0*zbot(2,2)*cdexp(-ag1-ag2)/zdenom
        call rotz(zrot,-bs,zprd)
c
c> Propagate all the parametric sensitivities of the deeper
c> than the current layer from the bottom to the top of the 
c> current layer
c
        do il=layer1,nl
          dzbot(1,1)=dzdal(il,1,1)
          dzbot(1,2)=dzdal(il,1,2)
          dzbot(2,1)=dzdal(il,2,1)
          dzbot(2,2)=dzdal(il,2,2)
          call zsprpg(dzbot,zbot,zrot,dz1,dz2,ag1,ag2,dztop)
          dzdalrot(il,1,1)=dztop(1,1)
          dzdalrot(il,1,2)=dztop(1,2)
          dzdalrot(il,2,1)=dztop(2,1)
          dzdalrot(il,2,2)=dztop(2,2)
c
          dzbot(1,1)=dzdat(il,1,1)
          dzbot(1,2)=dzdat(il,1,2)
          dzbot(2,1)=dzdat(il,2,1)
          dzbot(2,2)=dzdat(il,2,2)
          call zsprpg(dzbot,zbot,zrot,dz1,dz2,ag1,ag2,dztop)
          dzdatrot(il,1,1)=dztop(1,1)
          dzdatrot(il,1,2)=dztop(1,2)
          dzdatrot(il,2,1)=dztop(2,1)
          dzdatrot(il,2,2)=dztop(2,2)
c
          dzbot(1,1)=dzdbs(il,1,1)
          dzbot(1,2)=dzdbs(il,1,2)
          dzbot(2,1)=dzdbs(il,2,1)
          dzbot(2,2)=dzdbs(il,2,2)
          call zsprpg(dzbot,zbot,zrot,dz1,dz2,ag1,ag2,dztop)
          dzdbsrot(il,1,1)=dztop(1,1)
          dzdbsrot(il,1,2)=dztop(1,2)
          dzdbsrot(il,2,1)=dztop(2,1)
          dzdbsrot(il,2,2)=dztop(2,2)
c
          dzbot(1,1)=dzdh(il,1,1)
          dzbot(1,2)=dzdh(il,1,2)
          dzbot(2,1)=dzdh(il,2,1)
          dzbot(2,2)=dzdh(il,2,2)
          call zsprpg(dzbot,zbot,zrot,dz1,dz2,ag1,ag2,dztop)
          dzdhrot(il,1,1)=dztop(1,1)
          dzdhrot(il,1,2)=dztop(1,2)
          dzdhrot(il,2,1)=dztop(2,1)
          dzdhrot(il,2,2)=dztop(2,2)
        enddo         
c
c> Compute the parametric sensitivities with respect to the
c> parameters of the current layer
c
c> For the isotropic case
c
        if(layani(layer).eq.0.and.a1.eq.a2)then
          call zscua1(zbot,zrot,a1,dz1,dz2,ag1,ag2,dztop)
          dzdalrot(layer,1,1)=dztop(1,1)
          dzdalrot(layer,1,2)=dztop(1,2)
          dzdalrot(layer,2,1)=dztop(2,1)
          dzdalrot(layer,2,2)=dztop(2,2)
c
          call zscua2(zbot,zrot,a2,dz1,dz2,ag1,ag2,dztop)
          dzdatrot(layer,1,1)=dztop(1,1)
          dzdatrot(layer,1,2)=dztop(1,2)
          dzdatrot(layer,2,1)=dztop(2,1)
          dzdatrot(layer,2,2)=dztop(2,2)
c
          dzdbsrot(layer,1,1)=0.d0
          dzdbsrot(layer,1,2)=0.d0
          dzdbsrot(layer,2,1)=0.d0
          dzdbsrot(layer,2,2)=0.d0
c
          call zscuh(zbot,zrot,a1,a2,dz1,dz2,ag1,ag2,dztop)
          dzdhrot(layer,1,1)=dztop(1,1)
          dzdhrot(layer,1,2)=dztop(1,2)
          dzdhrot(layer,2,1)=dztop(2,1)
          dzdhrot(layer,2,2)=dztop(2,2)
c
c> For the general anisotropic case                
c
        else      
          call zscua1(zbot,zrot,a1,dz1,dz2,ag1,ag2,dztop)
          dzdalrot(layer,1,1)=dztop(1,1)
          dzdalrot(layer,1,2)=dztop(1,2)
          dzdalrot(layer,2,1)=dztop(2,1)
          dzdalrot(layer,2,2)=dztop(2,2)
c
          call zscua2(zbot,zrot,a2,dz1,dz2,ag1,ag2,dztop)
          dzdatrot(layer,1,1)=dztop(1,1)
          dzdatrot(layer,1,2)=dztop(1,2)
          dzdatrot(layer,2,1)=dztop(2,1)
          dzdatrot(layer,2,2)=dztop(2,2)
c
          call zscubs(zbot,zrot,dz1,dz2,ag1,ag2,dztop)
          dzdbsrot(layer,1,1)=dztop(1,1)
          dzdbsrot(layer,1,2)=dztop(1,2)
          dzdbsrot(layer,2,1)=dztop(2,1)
          dzdbsrot(layer,2,2)=dztop(2,2)
c
          call zscuh(zbot,zrot,a1,a2,dz1,dz2,ag1,ag2,dztop)
          dzdhrot(layer,1,1)=dztop(1,1)
          dzdhrot(layer,1,2)=dztop(1,2)
          dzdhrot(layer,2,1)=dztop(2,1)
          dzdhrot(layer,2,2)=dztop(2,2)
c
          layani(layer)=1
        endif
c
c> Set the reference direction to the anisotropy strike of
c> the current layer and go on to the next layer
c
        bsref=bs
c
      enddo
c
c> On the surface, rotate both the impedance and the parametric
c> sensitivities into the original coordinate system and return
c
      if(bsref.ne.0.d0)then
        call rotz(zrot,-bsref,z)
        call rotzs(dzdalrot,nl,1,-bsref,dzdal)
        call rotzs(dzdatrot,nl,1,-bsref,dzdat)
        call rotzs(dzdbsrot,nl,1,-bsref,dzdbs)
        call rotzs(dzdhrot,nl,1,-bsref,dzdh)
      else
        z(1,1)=zrot(1,1)
        z(1,2)=zrot(1,2)
        z(2,1)=zrot(2,1)
        z(2,2)=zrot(2,2)
        do ix=1,2
          do iy=1,2
            do il=1,nl
              dzdal(il,ix,iy)=dzdalrot(il,ix,iy)
              dzdat(il,ix,iy)=dzdatrot(il,ix,iy)
              dzdbs(il,ix,iy)=dzdbsrot(il,ix,iy)
              dzdh(il,ix,iy)=dzdhrot(il,ix,iy)
            enddo
          enddo
        enddo
      endif
c
      return
c
      end
c===> END SUBROUTINE ZS1ANEF
c
c
c===> SUBROUTINE ZS1ANIS
c     ================================================================
      subroutine zs1anis(layani,h,rop,ustr,udip,usla,al,at,blt,nl,
     &                   per,z,dzdal,dzdat,dzdbs,dzdh,
     &                   dzdsgpx,dzdsgpy,dzdsgpz,dzdstr,dzddip,dzdsla)
c     ================================================================
c     Computes, from the sensitivities of the impedance with respect
c     to the effective azimuthal anisotropy parameters, sensitivities
c     with respect to the true physical anisotropy parameters, i.e.,
c     with respect to the principal conductivities and elementary
c     anisotropy directions, strike, dip, and slant.
c
      real*8 pi
      parameter(pi=3.14159265358979323846264338327950288d0)
c
      parameter(nlmax=1001)
c
      real*8 h(nlmax),rop(nlmax,3),ustr(nlmax),udip(nlmax),usla(nlmax)
      real*8 al(nlmax),at(nlmax),blt(nlmax)
      integer*4 layani(nlmax)
      real*8 per
      complex*16 z(2,2)
      complex*16 dzdal(nlmax,2,2),dzdat(nlmax,2,2)
      complex*16 dzdbs(nlmax,2,2),dzdh(nlmax,2,2)
      complex*16 dzdsgpx(nlmax,2,2),dzdsgpy(nlmax,2,2)
      complex*16 dzdsgpz(nlmax,2,2),dzdstr(nlmax,2,2)
      complex*16 dzddip(nlmax,2,2),dzdsla(nlmax,2,2)
c
      real*8 hd,sgpx,sgpy,sgpz,rstr,rdip,rsla
      real*8 asg,bsg,csg,dsg,dbsgpz,dsgpxy
      real*8 cstr,sstr,cdip,sdip,csla,ssla
      real*8 c2str,s2str,c2dip,s2dip,c2sla,s2sla
      real*8 sigxx,sigxy,sigxz,sigyy,sigyz,sigzz,axx,axy,ayy
      real*8 dsigxxdsgpx,dsigxxdsgpy,dsigxxdsgpz
      real*8 dsigxxdstr,dsigxxddip,dsigxxdsla
      real*8 dsigxydsgpx,dsigxydsgpy,dsigxydsgpz
      real*8 dsigxydstr,dsigxyddip,dsigxydsla
      real*8 dsigxzdsgpx,dsigxzdsgpy,dsigxzdsgpz
      real*8 dsigxzdstr,dsigxzddip,dsigxzdsla
      real*8 dsigyydsgpx,dsigyydsgpy,dsigyydsgpz
      real*8 dsigyydstr,dsigyyddip,dsigyydsla
      real*8 dsigyzdsgpx,dsigyzdsgpy,dsigyzdsgpz
      real*8 dsigyzdstr,dsigyzddip,dsigyzdsla
      real*8 dsigzzdsgpx,dsigzzdsgpy,dsigzzdsgpz
      real*8 dsigzzdstr,dsigzzddip,dsigzzdsla
      real*8 daxxdsgpx,daxxdsgpy,daxxdsgpz,daxxdstr,daxxddip,daxxdsla
      real*8 dayydsgpx,dayydsgpy,dayydsgpz,dayydstr,dayyddip,dayydsla
      real*8 daxydsgpx,daxydsgpy,daxydsgpz,daxydstr,daxyddip,daxydsla
      real*8 p1,p2,p1a,p1b
      real*8 a12dif,c2bs,s2bs,axysum,axydif,dda1,dda2,ddbs
      real*8 dummy8
c
      do layer=1,nl
        hd=1.d3*dble(h(layer))
        sgpx=1.d0/dble(rop(layer,1))
        sgpy=1.d0/dble(rop(layer,2))
        sgpz=1.d0/dble(rop(layer,3))
c
        rstr=pi*dble(ustr(layer))/180.d0
        rdip=pi*dble(udip(layer))/180.d0
        rsla=pi*dble(usla(layer))/180.d0
c
        cstr=dcos(rstr)
        sstr=dsin(rstr)
        cdip=dcos(rdip)
        sdip=dsin(rdip)
        csla=dcos(rsla)
        ssla=dsin(rsla)
        c2str=dcos(2.d0*rstr)
        s2str=dsin(2.d0*rstr)
        c2dip=dcos(2.d0*rdip)
        s2dip=dsin(2.d0*rdip)
        c2sla=dcos(2.d0*rsla)
        s2sla=dsin(2.d0*rsla)
c
        dsgpxy=sgpx-sgpy
        asg=sgpx*csla**2.d0+sgpy*ssla**2.d0
        bsg=sgpx*ssla**2.d0+sgpy*csla**2.d0
        csg=bsg*cdip**2.d0+sgpz*sdip**2.d0
        dsg=bsg*sdip**2.d0+sgpz*cdip**2.d0
        dbsgpz=bsg-sgpz
c
c> sigxx and its derivatives
c
        sigxx=asg*cstr**2.d0+csg*sstr**2.d0-
     &        0.5d0*dsgpxy*cdip*s2str*s2sla
c
        dsigxxdsgpx=(csla*cstr-ssla*cdip*sstr)**2.d0
        dsigxxdsgpy=(ssla*cstr+csla*cdip*sstr)**2.d0
        dsigxxdsgpz=(sdip*sstr)**2.d0
        dsigxxdstr=(-asg+csg)*s2str-dsgpxy*cdip*s2sla*c2str
        dsigxxddip=-dbsgpz*s2dip*sstr**2.d0+
     &             0.5d0*dsgpxy*sdip*s2sla*s2str
        dsigxxdsla=dsgpxy*((sstr*cdip)**2.d0*s2sla-cstr**2.d0*s2sla-
     &             cdip*s2str*c2sla)
c
c> sigxy and its derivatives
c
        sigxy=0.5d0*(asg-csg)*s2str+0.5d0*dsgpxy*cdip*c2str*s2sla
c
        dsigxydsgpx=0.5d0*((csla**2.d0-(ssla*cdip)**2.d0)*s2str+
     &              cdip*c2str*s2sla)
        dsigxydsgpy=0.5d0*((ssla**2.d0-(csla*cdip)**2.d0)*s2str-
     &              cdip*c2str*s2sla)
        dsigxydsgpz=-0.5d0*sdip**2.d0*s2str
        dsigxydstr=(asg-csg)*c2str-dsgpxy*cdip*s2str*s2sla
        dsigxyddip=0.5d0*dbsgpz*s2dip*s2str-
     &             0.5d0*dsgpxy*sdip*c2str*s2sla
        dsigxydsla=0.5d0*dsgpxy*(2.d0*cdip*c2str*c2sla-
     &             (1.d0+cdip**2.d0)*s2str*s2sla)
c
c> sigxz and its derivatives
c
        sigxz=0.5d0*dsgpxy*sdip*cstr*s2sla-0.5d0*dbsgpz*s2dip*sstr
c
        dsigxzdsgpx=(cstr*csla-sstr*cdip*ssla)*ssla*sdip
        dsigxzdsgpy=-(cstr*ssla+sstr*cdip*csla)*csla*sdip
        dsigxzdsgpz=0.5d0*s2dip*sstr
        dsigxzdstr=-0.5d0*dsgpxy*sdip*sstr*s2sla-
     &             0.5d0*dbsgpz*s2dip*cstr
        dsigxzddip=0.5d0*dsgpxy*cdip*cstr*s2sla-
     &             dbsgpz*c2dip*sstr
        dsigxzdsla=0.5d0*dsgpxy*(2.d0*sdip*cstr*c2sla-s2dip*sstr*s2sla)
c
c> sigyy and its derivatives
c
        sigyy=asg*sstr**2.d0+csg*cstr**2.d0+
     &        0.5d0*dsgpxy*cdip*s2str*s2sla
c
        dsigyydsgpx=(sstr*csla+cdip*cstr*ssla)**2.d0
        dsigyydsgpy=(sstr*ssla-cdip*cstr*csla)**2.d0
        dsigyydsgpz=(cstr*sdip)**2.d0
        dsigyydstr=(asg-csg)*s2str+dsgpxy*cdip*c2str*s2sla
        dsigyyddip=-dbsgpz*s2dip*cstr**2.d0-
     &             0.5d0*dsgpxy*sdip*s2str*s2sla
        dsigyydsla=dsgpxy*(cdip*s2str*c2sla+(cdip*cstr)**2.d0*s2sla-
     &             sstr**2.d0*s2sla)
c
c> sigyz and its derivatives
c
        sigyz=0.5d0*dsgpxy*sdip*sstr*s2sla+0.5d0*dbsgpz*s2dip*cstr
c
        dsigyzdsgpx=(sstr*csla+cdip*cstr*ssla)*sdip*ssla
        dsigyzdsgpy=-(sstr*ssla-cdip*cstr*csla)*sdip*csla
        dsigyzdsgpz=-0.5d0*s2dip*cstr
        dsigyzdstr=0.5d0*dsgpxy*sdip*cstr*s2sla-
     &             0.5d0*dbsgpz*s2dip*sstr
        dsigyzddip=0.5d0*dsgpxy*cdip*sstr*s2sla+
     &             dbsgpz*c2dip*cstr
        dsigyzdsla=0.5d0*dsgpxy*(2.d0*sdip*sstr*c2sla+s2dip*cstr*s2sla)
c
c> sigzz and its derivatives
c
        sigzz=dsg
c
        dsigzzdsgpx=(sdip*ssla)**2.d0
        dsigzzdsgpy=(sdip*csla)**2.d0
        dsigzzdsgpz=cdip**2.d0
        dsigzzdstr=0.d0
        dsigzzddip=dbsgpz*s2dip
        dsigzzdsla=dsgpxy*s2sla*sdip**2.d0
c
c> axx and its derivatives
c
        axx=sigxx-sigxz*sigxz/sigzz
c
        p1=2.d0*sigxz/sigzz
        p2=(sigxz/sigzz)**2.d0
        daxxdsgpx=dsigxxdsgpx-p1*dsigxzdsgpx+p2*dsigzzdsgpx
        daxxdsgpy=dsigxxdsgpy-p1*dsigxzdsgpy+p2*dsigzzdsgpy
        daxxdsgpz=dsigxxdsgpz-p1*dsigxzdsgpz+p2*dsigzzdsgpz
        daxxdstr=dsigxxdstr-p1*dsigxzdstr+p2*dsigzzdstr
        daxxddip=dsigxxddip-p1*dsigxzddip+p2*dsigzzddip
        daxxdsla=dsigxxdsla-p1*dsigxzdsla+p2*dsigzzdsla
c
c> ayy and its derivatives
c
        ayy=sigyy-sigyz*sigyz/sigzz
c
        p1=2.d0*sigyz/sigzz
        p2=(sigyz/sigzz)**2.d0
        dayydsgpx=dsigyydsgpx-p1*dsigyzdsgpx+p2*dsigzzdsgpx
        dayydsgpy=dsigyydsgpy-p1*dsigyzdsgpy+p2*dsigzzdsgpy
        dayydsgpz=dsigyydsgpz-p1*dsigyzdsgpz+p2*dsigzzdsgpz
        dayydstr=dsigyydstr-p1*dsigyzdstr+p2*dsigzzdstr
        dayyddip=dsigyyddip-p1*dsigyzddip+p2*dsigzzddip
        dayydsla=dsigyydsla-p1*dsigyzdsla+p2*dsigzzdsla
c
c> axy and its derivatives
c
        axy=sigxy-sigxz*sigyz/sigzz
c
        p1a=sigxz/sigzz
        p1b=sigyz/sigzz
        p2=p1a*p1b
        daxydsgpx=dsigxydsgpx-p1a*dsigyzdsgpx-p1b*dsigxzdsgpx+
     &            p2*dsigzzdsgpx
        daxydsgpy=dsigxydsgpy-p1a*dsigyzdsgpy-p1b*dsigxzdsgpy+
     &            p2*dsigzzdsgpy
        daxydsgpz=dsigxydsgpz-p1a*dsigyzdsgpz-p1b*dsigxzdsgpz+
     &            p2*dsigzzdsgpz
        daxydstr=dsigxydstr-p1a*dsigyzdstr-p1b*dsigxzdstr+
     &            p2*dsigzzdstr
        daxyddip=dsigxyddip-p1a*dsigyzddip-p1b*dsigxzddip+
     &            p2*dsigzzddip
        daxydsla=dsigxydsla-p1a*dsigyzdsla-p1b*dsigxzdsla+
     &            p2*dsigzzdsla
c
        c2bs=dcos(2.d0*blt(layer))
        s2bs=dsin(2.d0*blt(layer))
        a12dif=al(layer)-at(layer)
c
c> dZ/d(sgpx)
c
        axysum=0.5d0*(daxxdsgpx+dayydsgpx)
        axydif=0.5d0*(daxxdsgpx-dayydsgpx)
        dda1=axysum+axydif*c2bs+daxydsgpx*s2bs
        dda2=axysum-axydif*c2bs-daxydsgpx*s2bs
c
c> Intrinsic function TINY(real_argument) (Microsoft/Compaq Fortran)
c> gives the smallest number in the model representing the same type 
c> and kind parameters as the argument. Alternatives are IMSL functions
c> AMACH(1) or DMACH(1) that give the smallest normalized positive 
c> number in the computer's single-precision or double-precision
c> arithmetic, respectively, or adequate functions for machine constants
c> determination on other systems
c
        if(dabs(a12dif).lt.tiny(a12dif))then
          ddbs=0.d0
        else
          ddbs=(daxydsgpx*c2bs-axydif*s2bs)/a12dif
        endif
        do ix=1,2
          do iy=1,2
            dzdsgpx(layer,ix,iy)=dzdal(layer,ix,iy)*dda1+
     &                           dzdat(layer,ix,iy)*dda2+
     &                           dzdbs(layer,ix,iy)*ddbs
          enddo
        enddo
c
c> dZ/d(sgpy)
c
        axysum=0.5d0*(daxxdsgpy+dayydsgpy)
        axydif=0.5d0*(daxxdsgpy-dayydsgpy)
        dda1=axysum+axydif*c2bs+daxydsgpy*s2bs
        dda2=axysum-axydif*c2bs-daxydsgpy*s2bs
        if(dabs(a12dif).lt.tiny(a12dif))then
          ddbs=0.d0
        else
          ddbs=(daxydsgpy*c2bs-axydif*s2bs)/a12dif
        endif
        do ix=1,2
          do iy=1,2
            dzdsgpy(layer,ix,iy)=dzdal(layer,ix,iy)*dda1+
     &                           dzdat(layer,ix,iy)*dda2+
     &                           dzdbs(layer,ix,iy)*ddbs
          enddo
        enddo
c
c> dZ/d(sgpz)
c
        axysum=0.5d0*(daxxdsgpz+dayydsgpz)
        axydif=0.5d0*(daxxdsgpz-dayydsgpz)
        dda1=axysum+axydif*c2bs+daxydsgpz*s2bs
        dda2=axysum-axydif*c2bs-daxydsgpz*s2bs
        if(dabs(a12dif).lt.tiny(a12dif))then
          ddbs=0.d0
        else
          ddbs=(daxydsgpz*c2bs-axydif*s2bs)/a12dif
        endif
        do ix=1,2
          do iy=1,2
            dzdsgpz(layer,ix,iy)=dzdal(layer,ix,iy)*dda1+
     &                           dzdat(layer,ix,iy)*dda2+
     &                           dzdbs(layer,ix,iy)*ddbs
          enddo
        enddo
c
c> dZ/d(strike)
c
        axysum=0.5d0*(daxxdstr+dayydstr)
        axydif=0.5d0*(daxxdstr-dayydstr)
        dda1=axysum+axydif*c2bs+daxydstr*s2bs
        dda2=axysum-axydif*c2bs-daxydstr*s2bs
        if(dabs(a12dif).lt.tiny(a12dif))then
          ddbs=0.d0
        else
          ddbs=(daxydstr*c2bs-axydif*s2bs)/a12dif
        endif
        do ix=1,2
          do iy=1,2
            dzdstr(layer,ix,iy)=dzdal(layer,ix,iy)*dda1+
     &                          dzdat(layer,ix,iy)*dda2+
     &                          dzdbs(layer,ix,iy)*ddbs
          enddo
        enddo
c
c> dZ/d(dip)
c
        axysum=0.5d0*(daxxddip+dayyddip)
        axydif=0.5d0*(daxxddip-dayyddip)
        dda1=axysum+axydif*c2bs+daxyddip*s2bs
        dda2=axysum-axydif*c2bs-daxyddip*s2bs
        if(dabs(a12dif).lt.tiny(a12dif))then
          ddbs=0.d0
        else
          ddbs=(daxyddip*c2bs-axydif*s2bs)/a12dif
        endif
        do ix=1,2
          do iy=1,2
            dzddip(layer,ix,iy)=dzdal(layer,ix,iy)*dda1+
     &                          dzdat(layer,ix,iy)*dda2+
     &                          dzdbs(layer,ix,iy)*ddbs
          enddo
        enddo
c
c> dZ/d(slant)
c
        axysum=0.5d0*(daxxdsla+dayydsla)
        axydif=0.5d0*(daxxdsla-dayydsla)
        dda1=axysum+axydif*c2bs+daxydsla*s2bs
        dda2=axysum-axydif*c2bs-daxydsla*s2bs
        if(dabs(a12dif).lt.tiny(a12dif))then
          ddbs=0.d0
        else
          ddbs=(daxydsla*c2bs-axydif*s2bs)/a12dif
        endif
        do ix=1,2
          do iy=1,2
            dzdsla(layer,ix,iy)=dzdal(layer,ix,iy)*dda1+
     &                          dzdat(layer,ix,iy)*dda2+
     &                          dzdbs(layer,ix,iy)*ddbs
          enddo
        enddo
c
      enddo
c
      return
      end
c===> END SUBROUTINE ZS1ANIS
c
c
c===> SUBROUTINE ROTZ
c     =============================
      subroutine rotz(za,betrad,zb)
c     =============================
c     Rotates the impedance ZA by BETRAD (in radians) to obtain ZB
c
      real*8 betrad
      complex*16 za(2,2),zb(2,2)
c
      real*8 co2,si2
      complex*16 sum1,sum2,dif1,dif2
c
      co2=dcos(2.d0*betrad)
      si2=dsin(2.d0*betrad)
c
      sum1=za(1,1)+za(2,2)
      sum2=za(1,2)+za(2,1)
      dif1=za(1,1)-za(2,2)
      dif2=za(1,2)-za(2,1)
c
      zb(1,1)=0.5d0*(sum1+dif1*co2+sum2*si2)
      zb(1,2)=0.5d0*(dif2+sum2*co2-dif1*si2)
      zb(2,1)=0.5d0*(-dif2+sum2*co2-dif1*si2)
      zb(2,2)=0.5d0*(sum1-dif1*co2-sum2*si2)
c
      return
      end
c===> END SUBROUTINE ROTZ
c
c
c===> SUBROUTINE ROTZS
c     =======================================
      subroutine rotzs(dza,nla,la,betrad,dzb)
c     =======================================
c     Rotates the sensitivities DZA of layers from LA to NLA by 
c     BETRAD (in radians) to obtain DZB
c
      parameter(nlmax=1001)
      real*8 betrad
      complex*16 dza(nlmax,2,2),dzb(nlmax,2,2)
c
      real*8 co2,si2
      complex*16 sum1,sum2,dif1,dif2
c
      co2=dcos(2.d0*betrad)
      si2=dsin(2.d0*betrad)
c
      do l=la,nla
        sum1=dza(l,1,1)+dza(l,2,2)
        sum2=dza(l,1,2)+dza(l,2,1)
        dif1=dza(l,1,1)-dza(l,2,2)
        dif2=dza(l,1,2)-dza(l,2,1)
c
        dzb(l,1,1)=0.5d0*(sum1+dif1*co2+sum2*si2)
        dzb(l,1,2)=0.5d0*(dif2+sum2*co2-dif1*si2)
        dzb(l,2,1)=0.5d0*(-dif2+sum2*co2-dif1*si2)
        dzb(l,2,2)=0.5d0*(sum1-dif1*co2-sum2*si2)
      enddo
c
      return
      end
c===> END SUBROUTINE ROTZS
c
c
c===> SUBROUTINE CPANIS
c     =====================================================
      subroutine cpanis(rop,ustr,udip,usla,nl,sg,al,at,blt)
c     =====================================================
c     Computes effective azimuthal anisotropy parameters in 
c     anisotropic layers from their principal resistivities
c     and elementary anisotropic directions
c
c     Input:
c     ------
c     ROP(NLMAX,3).....real*8 array of principal resistivities
c                      of the anisotropic layers in Ohm*m
c     USTR(NLMAX)......real*8 array with the anisotropy strike
c                      (first Euler's rotation) of the layers 
c                      in degrees
c     UDIP(NLMAX)......real*8 array with the anisotropy dip
c                      (second Euler's rotation) of the layers 
c                      in degrees
c     USLA(NLMAX)......real*8 array with the anisotropy slant
c                      (third Euler's rotation) of the layers 
c                      in degrees
c     NL...............integer*4 number of layers in the model
c                      including the basement, NL<=NLMAX
c
c     Output:
c     -------
c     SG(NLMAX,3,3)....real*8 array with conductivity tensor
c                      elements of the individual layers of the
c                      anisotropic model in S/m. The tensor is
c                      always symmetric and positive definite
c     AL(NLMAX)........real*8 array with the maximum effective
c                      horizontal conductivities of the layers 
c                      in S/m
c     AT(NLMAX)........real*8 array with the minimum effective
c                      horizontal conductivities of the layers 
c                      in S/m
c     BLT(NLMAX).......real*8 array with the effective horizontal
c                      anisotropy strike (direction of the maximum
c                      conductivity) of the layers in RADIANS
c
      real*8 pi
      parameter(pi=3.14159265358979323846264338327950288d0)
c
      parameter(nlmax=1001)
c
      real*8 rop(nlmax,3),ustr(nlmax),udip(nlmax),usla(nlmax)
      real*8 sgp(nlmax,3),sg(nlmax,3,3)
      real*8 al(nlmax),at(nlmax),blt(nlmax)
c
      real*8 rstr,rdip,rsla,sps,cps,sth,cth,sfi,cfi
      real*8 pom1,pom2,pom3,c2ps,s2ps,c2th,s2th,csps,csth
      real*8 axx,axy,ayx,ayy,da12
c
      do layer=1,nl
c
        do j=1,3
          sgp(layer,j)=1.d0/dble(rop(layer,j))
        enddo
c
        rstr=pi*dble(ustr(layer))/180.d0
        rdip=pi*dble(udip(layer))/180.d0
        rsla=pi*dble(usla(layer))/180.d0
        sps=dsin(rstr)
        cps=dcos(rstr)
        sth=dsin(rdip)
        cth=dcos(rdip)
        sfi=dsin(rsla)
        cfi=dcos(rsla)
        pom1=sgp(layer,1)*cfi*cfi+sgp(layer,2)*sfi*sfi
        pom2=sgp(layer,1)*sfi*sfi+sgp(layer,2)*cfi*cfi
        pom3=(sgp(layer,1)-sgp(layer,2))*sfi*cfi
        c2ps=cps*cps
        s2ps=sps*sps
        c2th=cth*cth
        s2th=sth*sth
        csps=cps*sps
        csth=cth*sth
c
c> Conductivity tensor
c
        sg(layer,1,1)=pom1*c2ps+pom2*s2ps*c2th-2.*pom3*cth*csps+
     &                sgp(layer,3)*s2th*s2ps
        sg(layer,1,2)=pom1*csps-pom2*c2th*csps+pom3*cth*(c2ps-s2ps)-
     &                sgp(layer,3)*s2th*csps
        sg(layer,1,3)=-pom2*csth*sps+pom3*sth*cps+sgp(layer,3)*csth*sps
        sg(layer,2,1)=sg(layer,1,2)
        sg(layer,2,2)=pom1*s2ps+pom2*c2ps*c2th+2.*pom3*cth*csps+
     &                sgp(layer,3)*s2th*c2ps
        sg(layer,2,3)=pom2*csth*cps+pom3*sth*sps-sgp(layer,3)*csth*cps
        sg(layer,3,1)=sg(layer,1,3)
        sg(layer,3,2)=sg(layer,2,3)
        sg(layer,3,3)=pom2*s2th+sgp(layer,3)*c2th
c
c> Effective horizontal 2*2 conductivity tensor
c
        axx=sg(layer,1,1)-sg(layer,1,3)*sg(layer,3,1)/sg(layer,3,3)
        axy=sg(layer,1,2)-sg(layer,1,3)*sg(layer,3,2)/sg(layer,3,3)
        ayx=sg(layer,2,1)-sg(layer,3,1)*sg(layer,2,3)/sg(layer,3,3)
        ayy=sg(layer,2,2)-sg(layer,2,3)*sg(layer,3,2)/sg(layer,3,3)
c
c> Principal conductivities and anisotropy strike of the effective
c> horizontal conductivity tensor
c
        da12=dsqrt((axx-ayy)*(axx-ayy)+4.d0*axy*ayx)
        al(layer)=0.5d0*(axx+ayy+da12)
        at(layer)=0.5d0*(axx+ayy-da12)
        if(da12.ge.tiny(da12))then
          blt(layer)=(axx-ayy)/da12
          blt(layer)=0.5d0*dacos(blt(layer))
        else
          blt(layer)=0.d0
        endif
        if(axy.lt.0.d0)blt(layer)=-blt(layer)
c
      enddo
c
      return
      end
c===> END SUBROUTINE CPANIS
c
c
c===> REAL*8 FUNCTION DPHASE
c     ===========================
      real*8 function dphase(z16)
c     ===========================
c     Computes the real*8 phase of a complex*16 value Z16
c
      real*8 pi
      parameter(pi=3.14159265358979323846264338327950288d0)
c
      complex*16 z16
      real*8 pom
      real*8 mytiny, tiny

      mytiny = 1.d-32
c
      if(dabs(dreal(z16)).ge.mytiny)then
        pom=datan(dimag(z16)/dreal(z16))
        if(dreal(z16).lt.0.d0)then
          if(dimag(z16).ge.0.d0)then
            dphase=pom+pi
          else
            dphase=pom-pi
          endif
        else
          dphase=pom
        endif
      else
        if(dimag(z16).lt.mytiny)then
          dphase=0.d0
        else
          if(dimag(z16).gt.0.d0)then
            dphase=5.d-1*pi
          else
            dphase=-5.d-1*pi
          endif
        endif
      endif
c
      return
      end
c===> END REAL*8 FUNCTION DPHASE
c
c
c===> SUBROUTINE ZSPRPG
c     ========================================================
      subroutine zsprpg(dzbot,zbot,ztop,dz1,dz2,ag1,ag2,dztop)
c     ========================================================
c     Propagates the parametric sensitivity from the bottom (DZBOT)
c     to the top (DZTOP) of an anisotropic layer
c
      real*8 pi
      parameter(pi=3.14159265358979323846264338327950288d0)
      complex*16 ic
      parameter(ic=(0.d0,1.d0))
c
      complex*16 dzbot(2,2),dztop(2,2),zbot(2,2),ztop(2,2)
      complex*16 dz1,dz2,ag1,ag2
c
      complex*16 zdenom
      complex*16 dzdenom,dn11,dn12,dn21,dn22,dtzbot,ddtzbot
c
      complex*16 dfm,dfp
c
      dtzbot=zbot(1,1)*zbot(2,2)-zbot(1,2)*zbot(2,1)
      zdenom=dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2)+
     &       zbot(1,2)*dfm(ag1)*dfp(ag2)/dz1-
     &       zbot(2,1)*dfp(ag1)*dfm(ag2)/dz2+
     &       dfp(ag1)*dfp(ag2)
      ddtzbot=dzbot(1,1)*zbot(2,2)+zbot(1,1)*dzbot(2,2)-
     &        dzbot(1,2)*zbot(2,1)-zbot(1,2)*dzbot(2,1)
      dzdenom=ddtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2)+
     &        dzbot(1,2)*dfm(ag1)*dfp(ag2)/dz1-
     &        dzbot(2,1)*dfp(ag1)*dfm(ag2)/dz2
c
      dn11=4.d0*dzbot(1,1)*cdexp(-ag1-ag2)
      dn12=dzbot(1,2)*dfp(ag1)*dfp(ag2)-
     &     dzbot(2,1)*dfm(ag1)*dfm(ag2)*dz1/dz2+
     &     ddtzbot*dfp(ag1)*dfm(ag2)/dz2
      dn21=dzbot(2,1)*dfp(ag1)*dfp(ag2)-
     &     dzbot(1,2)*dfm(ag1)*dfm(ag2)*dz2/dz1-
     &     ddtzbot*dfm(ag1)*dfp(ag2)/dz1
      dn22=4.d0*dzbot(2,2)*cdexp(-ag1-ag2)
c
      dztop(1,1)=(dn11-ztop(1,1)*dzdenom)/zdenom
      dztop(1,2)=(dn12-ztop(1,2)*dzdenom)/zdenom
      dztop(2,1)=(dn21-ztop(2,1)*dzdenom)/zdenom
      dztop(2,2)=(dn22-ztop(2,2)*dzdenom)/zdenom
c
      return
      end
c===> END SUBROUTINE ZSPRPG
c
c
c===> SUBROUTINE ZSCUA1   
c     =====================================================
      subroutine zscua1(zbot,ztop,a1,dz1,dz2,ag1,ag2,dztop)
c     =====================================================
c     Computes the derivative of the impedance tensor with respect
c     to the maximum horizontal conductivity (A1=AL) within an
c     anisotropic layer
c
      complex*16 dztop(2,2),zbot(2,2),ztop(2,2)
      complex*16 dz1,dz2,ag1,ag2
      real*8 a1
c
      complex*16 zdenom
      complex*16 dzdenom,dn12,dn21,dtzbot
      complex*16 dapom,dbpom,dcpom,ddpom,depom,dfpom,dgpom
c
      complex*16 dfm,dfp
c
      dtzbot=zbot(1,1)*zbot(2,2)-zbot(1,2)*zbot(2,1)
      zdenom=dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2)+
     &       zbot(1,2)*dfm(ag1)*dfp(ag2)/dz1-
     &       zbot(2,1)*dfp(ag1)*dfm(ag2)/dz2+
     &       dfp(ag1)*dfp(ag2)
c
      dapom=dtzbot*dfm(ag2)/dz2+zbot(1,2)*dfp(ag2)
      dbpom=dfp(ag2)-zbot(2,1)*dfm(ag2)/dz2
      dcpom=zbot(2,1)*dfp(ag2)-dz2*dfm(ag2)
      ddpom=dtzbot*dfp(ag2)+dz2*zbot(1,2)*dfm(ag2)
      depom=ag1*dfm(ag1)
      dfpom=(dfm(ag1)+ag1*dfp(ag1))/dz1
      dgpom=(dfm(ag1)-ag1*dfp(ag1))*dz1
c
      dzdenom=dfpom*dapom+depom*dbpom
      dn12=depom*dapom-dgpom*dbpom
      dn21=depom*dcpom-dfpom*ddpom
c
      dztop(1,1)=-0.5d0*ztop(1,1)*dzdenom/(a1*zdenom)
      dztop(1,2)=0.5d0*(dn12-ztop(1,2)*dzdenom)/(a1*zdenom)
      dztop(2,1)=0.5d0*(dn21-ztop(2,1)*dzdenom)/(a1*zdenom)
      dztop(2,2)=-0.5d0*ztop(2,2)*dzdenom/(a1*zdenom)
c
      return
      end
c===> END SUBROUTINE ZSCUA1
c
c
c===> SUBROUTINE ZSCUA2
c     =====================================================
      subroutine zscua2(zbot,ztop,a2,dz1,dz2,ag1,ag2,dztop)
c     =====================================================
c     Computes the derivative of the impedance tensor with respect
c     to the minimum horizontal conductivity (A2=AT) within an
c     anisotropic layer
c
      complex*16 dztop(2,2),zbot(2,2),ztop(2,2)
      complex*16 dz1,dz2,ag1,ag2
      real*8 a2
c
      complex*16 zdenom
      complex*16 dzdenom,dn12,dn21,dtzbot
      complex*16 dapom,dbpom,dcpom,ddpom,depom,dfpom,dgpom
c
      complex*16 dfm,dfp
c
      dtzbot=zbot(1,1)*zbot(2,2)-zbot(1,2)*zbot(2,1)
      zdenom=dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2)+
     &       zbot(1,2)*dfm(ag1)*dfp(ag2)/dz1-
     &       zbot(2,1)*dfp(ag1)*dfm(ag2)/dz2+
     &       dfp(ag1)*dfp(ag2)
c
      dapom=dtzbot*dfm(ag1)/dz1-zbot(2,1)*dfp(ag1)
      dbpom=dfp(ag1)+zbot(1,2)*dfm(ag1)/dz1
      dcpom=zbot(1,2)*dfp(ag1)+dz1*dfm(ag1)
      ddpom=dtzbot*dfp(ag1)-dz1*zbot(2,1)*dfm(ag1)
      depom=ag2*dfm(ag2)
      dfpom=(dfm(ag2)+ag2*dfp(ag2))/dz2
      dgpom=(dfm(ag2)-ag2*dfp(ag2))*dz2
c
      dzdenom=dfpom*dapom+depom*dbpom
      dn12=depom*dcpom+dfpom*ddpom
      dn21=-depom*dapom+dgpom*dbpom
c
      dztop(1,1)=-0.5d0*ztop(1,1)*dzdenom/(a2*zdenom)
      dztop(1,2)=0.5d0*(dn12-ztop(1,2)*dzdenom)/(a2*zdenom)
      dztop(2,1)=0.5d0*(dn21-ztop(2,1)*dzdenom)/(a2*zdenom)
      dztop(2,2)=-0.5d0*ztop(2,2)*dzdenom/(a2*zdenom)
c
      return
      end
c===> END SUBROUTINE ZSCUA2
c
c
c===> SUBROUTINE ZSCUBS     
c     ==================================================
      subroutine zscubs(zbot,ztop,dz1,dz2,ag1,ag2,dztop)
c     ==================================================
c     Computes the derivative of the impedance tensor with respect
c     to the effective horizontal anisotropy strike of the current
c     layer in the coordinate system aligned with that strike
c     direction
c
      complex*16 zbot(2,2),dztop(2,2),ztop(2,2)
      complex*16 dz1,dz2,ag1,ag2
c
      complex*16 dzbot(2,2),dtzbot,zdenom,dpom
c
      complex*16 dfm,dfp
c
      dztop(1,1)=-ztop(1,2)-ztop(2,1)
      dztop(1,2)=ztop(1,1)-ztop(2,2)
      dztop(2,1)=dztop(1,2)
      dztop(2,2)=-dztop(1,1)
c
      dtzbot=zbot(1,1)*zbot(2,2)-zbot(1,2)*zbot(2,1)
      zdenom=dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2)+
     &       zbot(1,2)*dfm(ag1)*dfp(ag2)/dz1-
     &       zbot(2,1)*dfp(ag1)*dfm(ag2)/dz2+
     &       dfp(ag1)*dfp(ag2)
c
      dzbot(1,1)=4.d0*(zbot(1,2)+zbot(2,1))*cdexp(-ag1-ag2)
      dzbot(1,2)=(zbot(1,1)-zbot(2,2))*
     &           (dfm(ag1)*dfm(ag2)*dz1/dz2-dfp(ag1)*dfp(ag2))
      dzbot(2,1)=(zbot(1,1)-zbot(2,2))*
     &           (dfm(ag1)*dfm(ag2)*dz2/dz1-dfp(ag1)*dfp(ag2))
      dzbot(2,2)=-4.d0*(zbot(1,2)+zbot(2,1))*cdexp(-ag1-ag2)
c
      dpom=(zbot(1,1)-zbot(2,2))*
     &     (dfm(ag1)*dfp(ag2)/dz1-dfp(ag1)*dfm(ag2)/dz2)
c
      dztop(1,1)=dztop(1,1)+(dzbot(1,1)+dpom*ztop(1,1))/zdenom
      dztop(1,2)=dztop(1,2)+(dzbot(1,2)+dpom*ztop(1,2))/zdenom
      dztop(2,1)=dztop(2,1)+(dzbot(2,1)+dpom*ztop(2,1))/zdenom
      dztop(2,2)=dztop(2,2)+(dzbot(2,2)+dpom*ztop(2,2))/zdenom
c      
      return
      end
c===> END SUBROUTINE ZSCUBS

c===> SUBROUTINE ZSCUH
c     =======================================================
      subroutine zscuh(zbot,ztop,a1,a2,dz1,dz2,ag1,ag2,dztop)
c     =======================================================
c     Computes the derivative of the impedance tensor with respect
c     to the thickness (in m) of the current anisotropic layer
c
      complex*16 dztop(2,2),zbot(2,2),ztop(2,2)
      complex*16 dz1,dz2,ag1,ag2
      real*8 a1,a2
c
      complex*16 zdenom
      complex*16 dzdenom,dn12,dn21,dtzbot
      complex*16 dapom1,dbpom1,dcpom1,ddpom1
      complex*16 dapom2,dbpom2,dcpom2,ddpom2
      complex*16 k1,k2
c
      complex*16 dfm,dfp
c
      dtzbot=zbot(1,1)*zbot(2,2)-zbot(1,2)*zbot(2,1)
      zdenom=dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2)+
     &       zbot(1,2)*dfm(ag1)*dfp(ag2)/dz1-
     &       zbot(2,1)*dfp(ag1)*dfm(ag2)/dz2+
     &       dfp(ag1)*dfp(ag2)
c
      k1=a1*dz1
      k2=a2*dz2
c
      dapom1=dtzbot*dfm(ag1)/dz1-zbot(2,1)*dfp(ag1)
      dbpom1=dfp(ag1)+zbot(1,2)*dfm(ag1)/dz1
      dcpom1=zbot(1,2)*dfp(ag1)+dz1*dfm(ag1)
      ddpom1=dtzbot*dfp(ag1)-dz1*zbot(2,1)*dfm(ag1)
c
      dapom2=dtzbot*dfm(ag2)/dz2+zbot(1,2)*dfp(ag2)
      dbpom2=dfp(ag2)-zbot(2,1)*dfm(ag2)/dz2
      dcpom2=zbot(2,1)*dfp(ag2)-dz2*dfm(ag2)
      ddpom2=dtzbot*dfp(ag2)+dz2*zbot(1,2)*dfm(ag2)
c
      dzdenom=k1*(dapom2*dfp(ag1)/dz1+dbpom2*dfm(ag1))+
     &        k2*(dapom1*dfp(ag2)/dz2+dbpom1*dfm(ag2))
      dn12=k1*(dapom2*dfm(ag1)+dbpom2*dz1*dfp(ag1))+
     &     k2*(dcpom1*dfm(ag2)+ddpom1*dfp(ag2)/dz2)
      dn21=k1*(dcpom2*dfm(ag1)-ddpom2*dfp(ag1)/dz1)-
     &     k2*(dapom1*dfm(ag2)+dbpom1*dz2*dfp(ag2))
c
      dztop(1,1)=-ztop(1,1)*dzdenom/zdenom
      dztop(1,2)=(dn12-ztop(1,2)*dzdenom)/zdenom
      dztop(2,1)=(dn21-ztop(2,1)*dzdenom)/zdenom
      dztop(2,2)=-ztop(2,2)*dzdenom/zdenom
c
      return
      end
c===> END SUBROUTINE ZSCUH     

c===> COMPLEX*16 FUNCTION DFM
c     ==========================
      complex*16 function dfm(x)
c     ==========================
c     Regularized hyperbolic sinus, dfm(x)=2.*sinh(x)/exp(x)
c
      complex*16 x
c
      dfm=1.d0-cdexp(-2.d0*x)
c
      return
      end
c===> END COMPLEX*16 FUNCTION DFM
c
c
c===> COMPLEX*16 FUNCTION DFP
c     ==========================
      complex*16 function dfp(x)
c     ==========================
c     Regularized hyperbolic cosinus, dfp(x)=2.*cosh(x)/exp(x)
c
      complex*16 x
c
      dfp=1.d0+cdexp(-2.d0*x)
c
      return
      end
c===> END COMPLEX*16 FUNCTION DFM

