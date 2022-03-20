! I/O routines for 3D_MT modelParam - extended Weerachai Siripunvaraporn's format

	!******************************************************************
	subroutine read_modelParam_ws(grid,m,cfile)

      ! opens cfile on unit ioPrm, writes out object of
      ! type modelParam in Weerachai Siripunvaraporn's format, closes file
      ! we can update the grid here, but the grid as an input is critical
      ! for setting the pointer to the grid in the modelParam

      type(grid_t), target, intent(inout)  :: grid
      type(modelParam_t), intent(out)	   :: m
      character(*), intent(in)             :: cfile
      integer		                       :: istat
      ! local variables

      type(rscalar)                        :: rho, ccond
      character(80)                        :: someChar='',paramType=''
      integer                              :: Nx, Ny, NzEarth, NzAir, someIndex=0, i, j, k
      real(8)                              :: alpha=3.

	  paramType = ''

      ! Open file and read grid
      open(unit=ioPrm, file=cfile, status='old', iostat=istat)

      ! First read the comment line
      read(ioPrm,'(a80)') someChar

	  ! Now read the second line with the grid dimensions
      read(ioPrm,'(a80)') someChar
      write(*,*) someChar
      read(someChar,*) Nx, Ny, NzEarth, someIndex

      if (someIndex /= 0) then
          write(0, *) 'Mapping not supported yet in read_modelParam_WS'
          stop
      end if

	  ! By default assume 'LINEAR RHO' - Weerachai's linear resistivity format
      if (index(someChar,'LOGE')>0) then
         paramType = 'LOGE'
      else if (index(someChar,'LOG10')>0) then
	  	 paramType = 'LOG10'
      else
         paramType = 'LINEAR'
      end if

	  ! No information about the air layers in file. Hardcoded here.
	  ! For backwards compatibility, keeping this logic for now...
	  ! but it may be overwritten by forward solver configuration file.
	  nzAir = 10

	  ! Initialize and read in the grid
      call create_grid(Nx,Ny,NzAir,NzEarth,grid)

      read(ioPrm,*) (grid%dx(j),j=1,grid%nx)
      read(ioPrm,*) (grid%dy(j),j=1,grid%ny)
      read(ioPrm,*) (grid%dz(j),j=grid%nzAir+1,grid%nzAir+grid%nzEarth)

      !   Following is Kush's approach to setting air layers:
      ! mirror imaging the dz values in the air layer with respect to
      ! earth layer as far as we can using the following formulation
      ! air layer(bottom:top) = (alpha)^(j-1) * earth layer(top:bottom)
      i = grid%nzAir+1
      j = 0
      do k = grid%nzAir, 1, -1
          j = j + 1
          grid%dz(k) = ((alpha)**(j-1))*grid%dz(i)
          i = i + 1
      end do

      ! the topmost air layer has to be at least 30 km
      if (grid%dz(1).lt.30000) then
          grid%dz(1) = 30000
      end if

	  ! Create and read in the resistivity values
      call create_rscalar(grid,rho,CELL_EARTH)

 	  ! Read the resistivity
 	  do k=1,nzEarth
 	  	do j=1,Ny
			read(ioPrm,*,iostat=istat) (rho%v(i,j,k), i=Nx,1,-1)
 	  	end do
 	  end do

 	  ! Note that our standard subroutine doesn't work with Weerachai's
 	  ! real value format. It is still better than either Mackie's or WS's...
      ! call read_rscalar(ioPrm,rho)

      ! Convert from conductivity to resistivity
      call copy_rscalar(ccond,rho)
      if ((index(paramType,'LOGE')>0) .or. (index(paramType,'LOG10')>0)) then
      	ccond%v = - rho%v
      else if (index(paramType,'LINEAR')>0) then
       	ccond%v = ONE/rho%v
 	  end if

      ! In case the grid origin is stored next (in metres!)...
      read(ioPrm,*,iostat=istat) grid%ox, grid%oy, grid%oz

	  ! Defaults to the grid centre at the Earth's surface
      if (istat /= 0) then
          grid%ox = -sum(grid%dx)/2.0
          grid%oy = -sum(grid%dy)/2.0
          grid%oz = 0.0
      endif

      read(ioPrm,*,iostat=istat) grid%rotdeg
      if (istat /= 0) then
          grid%rotdeg = 0.0
      end if

      close(ioPrm)

	  ! Finally create the model parameter
	  call create_modelParam(grid,paramType,m,ccond)

 	  ! In ModelSpace, save the user paramType for output
	  userParamType = paramType

	  ! ALWAYS convert modelParam to natural log for computations
	  paramType = 'LOGE'
	  call setType_modelParam(m,paramType)

	  ! now done with the rscalars, so deallocate
	  call deall_rscalar(rho)
	  call deall_rscalar(ccond)

	end subroutine read_modelParam_ws

	!******************************************************************
	subroutine write_modelParam_ws(m,cfile,comment)

      ! opens cfile on unit ioPrm, writes out object of
      ! type modelParam in Weerachai Siripunvaraporn's format, closes file

      type(modelParam_t), intent(in)	   :: m
      character(*), intent(in)             :: cfile
      character(*), intent(in), optional   :: comment
      integer           		           :: istat
      ! local variables
      type(grid_t)                         :: grid
      type(rscalar)                        :: rho,ccond
      character(80)                        :: paramType
      integer                              :: Nx, Ny, NzEarth, i, j, k

	  ! Convert modelParam to natural log or log10 for output
	  paramType = userParamType
	  call getValue_modelParam(m,paramType,ccond)

	  grid = ccond%grid

      open(unit=ioPrm, file=cfile, status='unknown', iostat=istat)

	  if (present(comment)) then
	  	write(ioPrm,*) '# ',trim(comment)
	  else
      	write(ioPrm,*) '# 3D MT model written by ModEM in WS format'
      end if

      ! Write grid geometry definitions
      Nx=grid%nx
      Ny=grid%ny
      NzEarth=grid%nz - grid%nzAir

      write(ioPrm,'(4i5)',advance='no') Nx,Ny,NzEarth,0
      write(ioPrm,*) trim(paramType)

      ! Write grid spacings
      do j=1,grid%nx
      	write(ioPrm,'(f12.3)',advance='no') grid%dx(j)
	  end do
	  write(ioPrm,*)
	  do j=1,grid%ny
      	write(ioPrm,'(f12.3)',advance='no') grid%dy(j)
	  end do
	  write(ioPrm,*)
	  do j=grid%nzAir+1,grid%nz
      	write(ioPrm,'(f12.3)',advance='no') grid%dz(j)
	  end do
	  write(ioPrm,*)

      ! Convert conductivity to resistivity
      call copy_rscalar(rho,ccond)
      if ((index(paramType,'LOGE')>0) .or. (index(paramType,'LOG10')>0)) then
      	rho%v = - ccond%v
      else if (index(paramType,'LINEAR')>0) then
       	rho%v = ONE/ccond%v
 	  end if

 	  ! Write the resistivity
 	  write(ioPrm,*)
 	  do k=1,nzEarth
 	  	do j=1,Ny
 	  		do i=Nx,1,-1
 	  			write(ioPrm,'(es13.5)',iostat=istat,advance='no') rho%v(i,j,k)
 	  		end do
 	  		write(ioPrm,*)
 	  	end do
 	  	write(ioPrm,*)
 	  end do

 	  ! Note that our standard subroutine doesn't work with Weerachai's
 	  ! real value format. It is still better than either Mackie's or WS's...
	  ! call write_rscalar(ioPrm,rho)

      ! Also write the grid origin (in metres!) and rotation (in degrees)...
      write(ioPrm,'(3f16.3)',iostat=istat) grid%ox, grid%oy, grid%oz
      write(ioPrm,'(f9.3)',iostat=istat)  grid%rotdeg
      close(ioPrm)

	  ! Now done with ccond, rho and grid, so deallocate
	  call deall_rscalar(rho)
	  call deall_rscalar(ccond)
	  call deall_grid(grid)

    end subroutine write_modelParam_ws