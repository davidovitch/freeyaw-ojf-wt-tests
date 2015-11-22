unit global;

interface
const
  maxwpset = 10;
  maxwplines = 100;
type
  array_3 = array[1..3] of single;
  array_4 = array[1..4] of single;
  array_15 = array[1..15] of single;
  array_25 = array[1..25] of single;

  Tlowpass2order=Record
    ksi : single;
    f0: single;
    Omf0: Extended;
    Omf1: Extended;
    Omf0_old: Extended;
    Omf1_old: Extended;
    stepno1:integer;
  end;

  Tfirstordervar=Record
    tau : single;
    x1,y1: single;
    x1_old,y1_old : single;
    stepno1 : integer;
  end;

  // record seems to be object oriented variables
  // se here you get Tavefiltervar.xvars, Tavefiltervar.time etc
  Tavefiltervar=Record
    xvars : array of double;
    time : single;
    size : integer;
    stepno1 : integer;
  end;

  Tpidvar=Record
    Kpro,Kdif,Kint  : single;
    outmin,outmax         : single;
    velmax                : single;

    error1                : single;
    outset1               : single;
    outres1               : single;
    stepno1               : integer;
    outset,outpro,outdif  : single;
    error1_old            : single;
    outset1_old           : single;
    outres1_old           : single;
    outres                : single;
  end;

  Twpdata = Record
     wpdata   : array[1..maxwplines,1..2] of single;
     lines    : integer;
  end;

  Tregulator = Record
     stepno    : integer;
     deltat    : single;
     wpname    : string;
     omega_ref_max : single;
     omega_ref : single;
     omega_min : single;
     omega_start : single;
     lambda    : single;
     rotor_radius : single;
     kga_set   : single;
     P_max      :single;
     T_max      :single;
     LSK      : single;
     const_power : boolean ;
     rel_limit :single;
     generator_stoptime : single;
     pitch_stopdelay : single;
     pitch_stopdelay2 : single;
     pitch_stopvelmax : single;
     pitch_stopvelmax2 : single;
     pitch_stoptype : integer;
     pitch_stopang : single;
     outputvektor_old : array_15;
     time_old  : single ;
     pit_limit : single;
     counter   : single;

     lowpass2order:Tlowpass2order;
     firstordervar1:Tfirstordervar;
     //omegafirstordervar: Tfirstordervar;
     omega2ordervar: Tlowpass2order;
     outmax2ordervar: Tlowpass2order;
     //omegaavefiltervar:Tavefiltervar;
     PID_com_var:Tpidvar;
     PID_gen_var:Tpidvar;
     OPdatavar:    Twpdata;
     wspfirstordervar: Tfirstordervar;
  end;

  Tgenerator = Record
     tau      : single;
     mgenwa1  : single;
     mgenera1 : single;
     stepno   : integer;
     deltat   : single;
     outputvektor_old :array_15;
     time_old : single;
  end;

  Tpitchservovar=record
    theta_ref  : single;
    theta_new  : single;
    theta      : single;
    theta_max  : single;
    theta_min  : single;
    pitvel_max :single;
    servofirstordervar : Tfirstordervar;
  end;

  Tglobal = Record
     theta    : array_3;
  end;

  Tbuttervar=Record
    butter_a :   array[1..3,1..3] of single;
    butter_b :   array[1..2,1..2] of single;
    inp,tem,oup,inp_old,tem_old,oup_old: array[1..3] of single;
    stepno1:integer;
  end;

const
  pi = 3.14159265359;
  degrad = 0.0174532925;
  raddeg = 57.2957795131;
var
  global_init : boolean=false;
  regulatorvar : Tregulator;
  generatorvar : Tgenerator;
  pitchservovar : array[1..3] of Tpitchservovar;
  wpdatavar     : Twpdata;
  globalvar     : Tglobal;
  datafil : textfile;
  firstordervar_gen:Tfirstordervar;
  pitchservofiltvar:Tfirstordervar;
  lowpass2ordergen:Tlowpass2order;
  lowpass2orderservo: Array[1..3] of Tlowpass2order;
  bw_var: Tbuttervar;
  bw_orig_var:Tbuttervar;
  bw_gen_var:Tbuttervar;
implementation

end.

