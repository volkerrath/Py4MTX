module sensMatrix

  use math_constants
  use file_units
  use utilities
  use DataSpace
  use ModelSpace

  use transmitters
  use dataTypes
  use receivers

  use dataio

  implicit none

  public 	:: create_sensMatrix, deall_sensMatrix
  public 	:: create_sensMatrixMTX, deall_sensMatrixMTX
  public    :: count_sensMatrixMTX
  public    :: write_sensMatrixMTX
  public    :: multBy_sensMatrixMTX

  !***********************************************************************
  ! Data type definitions for the full sensitivity matrix; a cross between
  ! a data vector and a model parameter to store the full matrix J = df/dm
  ! or the gradient of the data misfit, component-wise, i.e. (J^T)_i r_i.
  ! In this case, one column of J^T is a model parameter, so that J^T acts
  ! on a single data residual to produce the misfit gradient for that data
  ! point with respect to the model parameter; these can be summed up to
  ! obtain the full gradient. Can be useful as another measure of sensitivity.
  ! These matrices will be huge. We don't create them unless they are needed.

  ! basic data sensitivity vector for a single transmitter & data type
  type :: sensVector_t

      ! nComp is number of EM components observed (e.g., 2 (3) for
      ! MT (with verticl field TFs)) times 2 to allow for real/imag;
      ! has to match the data type index dt
      integer 		:: nComp = 0

      ! nSite is number of sites where these components are observed
      integer		:: nSite = 0

      ! sensitivities stored as model parameters; dimensions (nComp,nSite)
      type (modelParam_t), pointer, dimension(:,:) :: dm

      ! rx(nSite) is the array of indices into receiver dictionary
      integer, pointer, dimension(:) :: rx

      ! tx is index into transmitter dictionary
      integer		:: tx = 0

      ! dt is index into data type dictionary
      integer		:: dataType = 0

      logical       :: isComplex = .false.
      logical		:: allocated = .false.

      ! needed to avoid memory leaks for temporary function outputs
      logical		:: temporary = .false.

  end type sensVector_t


  ! sensMatrix is a collection of sensVector objects for all data types,
  ! *single* transmitter... full sensitivity matrix is an array of these object
  ! (avoids the extra layer of complication)
  type :: sensMatrix_t
      ! the number of dataVecs (generally the number of data types) associated
      ! with this transmitter (note: not the total number of data types)
      integer       :: ndt = 0

      ! array of sensVector's, usually one for each data type (dimension ndt)
      type (sensVector_t), pointer, dimension(:)   :: v

      ! tx is the index into transmitter dictionary
      integer       :: tx = 0

      logical       :: allocated = .false.

  end type sensMatrix_t

  type :: data_file_block

      ! this block of information constitutes user preferences about the data format;
      ! there is one entry per each transmitter type and data type... (iTxt,iDt)
      ! if there are multiple data blocks of the same transmitter & data types,
      ! the last value is used.
      character(200) :: info_in_file
      character(20)  :: sign_info_in_file
      integer        :: sign_in_file
      character(20)  :: units_in_file
      real           :: origin_in_file(2)
      real           :: geographic_orientation

     ! these lists contain the indices into the data vector for each data type;
     ! they make it possible to sort the data by receiver for output.
     ! no data denoted by zero index; dimensions (nTx) and (nTx,nRx).
     ! these indices are typically allocated as we read the data file
     integer, pointer, dimension(:)   :: tx_index
     integer, pointer, dimension(:)   :: dt_index
     integer, pointer, dimension(:,:) :: rx_index

     ! some transmitter types and data types don't go together
     logical         :: defined

  end type data_file_block

  type (data_file_block), pointer, save, private, dimension(:,:) :: fileInfo
    


Contains

  ! **********************************************************************
  ! creates an object of type sensMatrixTX:
  ! a vector containing sensitivities for a single transmitter.
  ! --> nComp: number of components, nSite: number of sites
  ! --> nDt: number of data types for transmitter tx

  subroutine create_sensMatrix(d, sigma0, sens)

    type (dataVector_t), intent(in)		 :: d
    type (modelParam_t), intent(in)		 :: sigma0
    type (sensMatrix_t), intent(inout)   :: sens
    ! local
    integer                              :: nComp,nSite,i,j,k,istat

    if(sens%allocated) then
       call errStop('input sens matrix already allocated in create_sensMatrix')
    else if(.not. d%allocated) then
       call errStop('input data vector not allocated in create_sensMatrix')
    end if

    sens%ndt = d%ndt

    allocate(sens%v(d%ndt), STAT=istat)

    do i = 1,d%nDt

       nComp = d%data(i)%nComp
       nSite = d%data(i)%nSite

       sens%v(i)%nComp = nComp
       sens%v(i)%nSite = nSite

       allocate(sens%v(i)%dm(nComp,nSite), STAT=istat)
       do j = 1,nComp
         do k = 1,nSite
         	sens%v(i)%dm(j,k) = sigma0
         end do
       end do

       sens%v(i)%tx = d%data(i)%tx
       allocate(sens%v(i)%rx(nSite), STAT=istat)
       sens%v(i)%rx = d%data(i)%rx

       sens%v(i)%dataType = d%data(i)%dataType
       sens%v(i)%isComplex = d%data(i)%isComplex

       sens%v(i)%allocated = .true.

    end do

    sens%tx = d%tx
    sens%allocated = .true.

  end subroutine create_sensMatrix

  !**********************************************************************
  ! deallocates all memory and reinitialized sensitivity matrix
  ! for one transmitter tx

  subroutine deall_sensMatrix(sens)

    type (sensMatrix_t)	:: sens
    ! local
    integer             	:: nComp,nSite,i,j,k,istat

    if(sens%allocated) then
       !  deallocate everything relevant
       do i = 1,sens%nDt

          sens%v(i)%tx = 0
          deallocate(sens%v(i)%rx, STAT=istat)

          sens%v(i)%dataType = 0
          sens%v(i)%isComplex = .false.

          do j = 1,nComp
            do k = 1,nSite
            	call deall_modelParam(sens%v(i)%dm(j,k))
            end do
          end do
          deallocate(sens%v(i)%dm, STAT=istat)

          sens%v(i)%nComp = 0
          sens%v(i)%nSite = 0
          sens%v(i)%allocated = .false.

       end do
       deallocate(sens%v, STAT=istat)
    end if

    sens%tx = 0
    sens%allocated = .false.

  end subroutine deall_sensMatrix


  ! **********************************************************************
  ! creates an object of type sensMatrixMTX:
  ! a vector containing sensitivities for all transmitters.

  subroutine create_sensMatrixMTX(d, sigma0, sens)

    type (dataVectorMTX_t), intent(in)  	:: d
    type (modelParam_t), intent(in)  	:: sigma0
    type (sensMatrix_t), pointer		:: sens(:)
    ! local
    integer                             :: nTx,i,istat

    if(associated(sens)) then
        call errStop('sensitivity matrix already allocated in create_sensMatrixMTX')
    end if

    nTx = d%nTx

    ! allocate space for the sensitivity matrix
    allocate(sens(nTx), STAT=istat)
    do i = 1,nTx
		call create_sensMatrix(d%d(i), sigma0, sens(i))
	end do

  end subroutine create_sensMatrixMTX


  !**********************************************************************
  ! deallocates all memory and reinitialized the full sensitivity matrix

  subroutine deall_sensMatrixMTX(sens)

    type (sensMatrix_t), pointer	:: sens(:)
    ! local
    integer             :: i,istat

    if(associated(sens)) then
       !  deallocate everything relevant
       do i = 1,size(sens)
          call deall_sensMatrix(sens(i))
       end do
       deallocate(sens, STAT=istat)
    endif

  end subroutine deall_sensMatrixMTX


  !**********************************************************************
  ! count all sensitivity values in the full sensitivity matrix

  function count_sensMatrixMTX(sens) result(N)

    type(sensMatrix_t), pointer	:: sens(:)
    integer				:: N
    ! local variables
    integer 				:: i,j,nTx

    nTx = size(sens)
    N = 0
    do i = 1,nTx
      do j = 1,sens(i)%nDt
        N = N + sens(i)%v(j)%nComp * sens(i)%v(j)%nSite
      enddo
    enddo

  end function count_sensMatrixMTX

  !*********************************************************************
  ! output is a quick fix, as always - reduces to nearly the same thing
  ! as before, a vector of model parameters
  subroutine write_sensMatrixMTX(sens,allData,cfile)

    type(sensMatrix_t), pointer	:: sens(:)
    type(dataVectorMTX_t), intent(in)         :: allData

    character(*), intent(in)				:: cfile
    ! local
    integer  iTx,iDt,iRx,nTx,nDt,nSite,nComp,icomp,i,j,k,l,istat,ios,nAll,iTxt
    real(8), allocatable            :: val(:) ! (ncomp)
    real(8), allocatable            :: err(:) ! (ncomp)
    ! logical, allocatable            :: exist(:) ! (ncomp)
    real(8) :: pTx, xRx(3)
    character(80) header, fmtstring
    character(40) sRx, cRx
    character(20) compid
    logical  :: SensWork
    character(200) tmp
    ! real(kind=prec)	:: xRx(3)
    ! real :: xRx(3)
    
    SensWork = .true.
	iTxt = 1   
    if(.not. associated(sens)) then
        call errStop('sensitivity matrix not allocated in write_sensMatrixMTX')
    end if

    open(unit=ioSens, file=cfile, form='unformatted', iostat=ios)
	write(0,*) 'Output sensitivity matrix...'


    write(header,*) 'Sensitivity Matrix'
	nAll = count_sensMatrixMTX(sens)
	write(ioSens) header
	write(ioSens) nAll

    nTx = size(sens)
    write(ioSens) nTx

    do i = 1,nTx
      nDt = sens(i)%nDt
      iTx = sens(i)%tx
      pTx= txDict(iTx)%period

      write(ioSens) nDt
      
      do j = 1,nDt
      
        nComp = sens(i)%v(j)%nComp
        nSite = sens(i)%v(j)%nSite
        iDt   = sens(i)%v(j)%dataType
        
        
!       SI_factor = ImpUnits(typeDict(iDt)%units,fileInfo(iTxt,iDt)%units_in_file)
        
        allocate(val(nComp),err(nComp),STAT=istat)
        ! write(*,*)
        write(ioSens) nSite

        do k = 1,nSite
            iRx = sens(i)%v(j)%rx(k)
            sRx = rxDict(iRx)%id
            xRx = rxDict(iRx)%x
 
            if (SensWork) then
                !write(*,*) 'Data Type: ', iDt, nComp, 
                select case (iDt)

                    !    Full_Impedance              = 1
                    !    Off_Diagonal_Impedance      = 2
                    !    Full_Vertical_Components    = 3
                    !    Full_Interstation_TF        = 4
                    !    Off_Diagonal_Rho_Phase      = 5
                    !    Phase_Tensor                = 6


                    case(1, 2, 3)
                        
                        fmtstring = '(g12.5,3x,a20,3f16.3,a8,3g16.6)'
                        val = allData%d(i)%data(j)%value(:,k)
                        err = allData%d(i)%data(j)%error(:,k)

                            
                        do icomp = 1,nComp/2
                            compid = typeDict(iDt)%id(icomp)
                            write(13,fmt=fmtstring) &
                                pTx,trim(sRx),xRx,trim(compid),&
                                val(2*icomp-1), val(2*icomp), err(2*icomp)
                        end do
                        
                        write(header,'(a,i10,a,i10,a,i10)') &
                            'Sensitivity for freq=',iTx,'; dataType=',iDt,'; site=',iRx

                        call writeVec_modelParam(nComp,sens(i)%v(j)%dm(:,k),header,cfile)                       


                    case(5, 6)
                            
                        fmtstring = '(g12.5,3x,a20,3f16.3,a8,2g16.6)'
                        val = allData%d(i)%data(j)%value(:,k)
                        err = allData%d(i)%data(j)%error(:,k)

                        do icomp = 1,nComp
                            compid = typeDict(iDt)%id(icomp)
                            write(13,fmt=fmtstring) &
                            pTx,trim(sRx),xRx,trim(compid),val(icomp),err(icomp)
                        end do
                        
                        write(header,'(a,i10,a,i10,a,i10)') &
                            'Sensitivity for freq=',iTx,'; dataType=',iDt,'; site=',iRx
                                
                        call writeVec_modelParam(nComp,sens(i)%v(j)%dm(:,k),header,cfile)
                    
                    end select


	        else
	           	write(header,'(a,i10,a,i10,a,i10)') &
					'Sensitivity for freq=',iTx,'; dataType=',iDt,'; site=',iRx
	                write(13,*) header
	        	call writeVec_modelParam(nComp,sens(i)%v(j)%dm(:,k),header,cfile)

        	end if

        end do ! rx
        
        deallocate(val,err,STAT=istat)
        
      end do  ! data type

    end do  ! tx

    close(ioSens)

  end subroutine write_sensMatrixMTX

  !*********************************************************************
  ! implements direct (real) matrix vector multiplication d = J m
  ! ... this can be used to test the implementation of calcJ,
  !     by comparing the output to that of Jmult
  subroutine multBy_sensMatrixMTX(sens,m,d)

    type(sensMatrix_t), pointer :: sens(:)
    type(modelParam_t), intent(in) :: m
    type(dataVectorMTX_t), intent(out) :: d
    ! local
    integer  nTx,nDt,nSite,iComp,nComp,i,j,k,istat
    logical  isComplex,errorBar

    if(.not. associated(sens)) then
        call errStop('sensitivity matrix not allocated in multBy_sensMatrixMTX')
    end if

    nTx = size(sens)
    call create_dataVectorMTX(nTx,d)

    do i = 1,nTx

      nDt = sens(i)%nDt
      call create_dataVector(nDt,d%d(i))

      do j = 1,nDt

        nComp = sens(i)%v(j)%nComp
        nSite = sens(i)%v(j)%nSite
        isComplex = sens(i)%v(j)%isComplex
        errorBar = .false.
        call create_dataBlock(nComp, nSite, d%d(i)%data(j), isComplex, errorBar)

        do k = 1,nSite

            do iComp = 1,nComp
                ! computes the dot products of the model parameters
                d%d(i)%data(j)%value(iComp,k) = dotProd_modelParam(sens(i)%v(j)%dm(iComp,k),m)

            end do ! components

        end do ! rx

        d%d(i)%data(j)%dataType = sens(i)%v(j)%dataType
        d%d(i)%data(j)%rx = sens(i)%v(j)%rx
        d%d(i)%data(j)%tx = sens(i)%v(j)%tx
        d%d(i)%data(j)%allocated = .true.

      end do  ! data type

      d%d(i)%tx = sens(i)%tx
      d%d(i)%allocated = .true.

    end do  ! tx

    d%allocated = .true.

  end subroutine multBy_sensMatrixMTX

end module sensMatrix
