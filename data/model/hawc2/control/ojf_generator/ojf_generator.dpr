library ojf_generator;
uses
  SysUtils,
  Classes,
  Dialogs,
  global in 'global.pas',
  procedures in 'procedures.pas';

Type
  array_1000 = array[0..12] of double;
Var
  // define variables used in init and update, these are constants
  K0            : double;
  t0            : double;
  K1            : double;
  t1            : double;
  K2            : double;
  T_min         : double;
  T_max         : double;
  filt_type     : double;
  lowpass2order : Tlowpass2order;
  firstordervar : Tfirstordervar;
  // variables for update which need to have an initial value
  deltat    : double;
  time_old  : double;
  stepno    : integer;
{$R *.res}


procedure init(var InputSignals : array_1000;
               var OutputSignals: array_1000); cdecl;
begin
  // duty cycle in the 3 time regions
  K0  :=Inputsignals[0];
  t0  :=Inputsignals[1];
  K1  :=Inputsignals[2];
  t1  :=Inputsignals[3];
  K2  :=Inputsignals[4];
  // limit the torque output, necessary to account for init instabilities etc
  T_min:=Inputsignals[5];
  T_max:=Inputsignals[6];
  // switch specifying witch filter to use
  filt_type:=Inputsignals[7];
  // filtering of the signal, 1st order case
  firstordervar.tau:=Inputsignals[8];
  // filtering of the signal, 2nd order case
  lowpass2order.f0:=Inputsignals[9];   // low pass cut-off freq
  lowpass2order.ksi:=Inputsignals[10]; //damping
  // initialize the other variables
  outputsignals[0]:=0.0;
  deltat:=0;
  time_old:=0.0;
  stepno:=0;
end;

procedure update(var InputSignals  : array_1000;
                 var OutputSignals : array_1000); cdecl;
var
  time      : double;
  rpm       : double;
  rpm_filt  : double;
  K         : double;
  Torque    : double;
begin
  time:=inputsignals[0];
  rpm :=inputsignals[1];

  // is the duty cycle changing? if so, adapt torque constant K
  if (time<=t0) then K:=K0
  else if (t0<time) and (time<=t1) then K:=K1
  else K:=K2;

  // only calculate these after the first step
  if time>time_old then
  begin
    deltat:=time-time_old;
    time_old:=time;
    inc(stepno);
  end;

  // filter the rpm signal according to the specified filter
  if (filt_type>1.9) then
     rpm_filt:=lowpass2orderfilt(deltat,stepno,lowpass2order,rpm)
  else if (filt_type>0.9) then
     rpm_filt:=firstorderfilter(deltat,stepno,firstordervar,rpm)
  // in all other cases, no filtering at all
  else
     rpm_filt:=rpm;

  Torque:=K*rpm_filt;

  // clip the torque signal if out of bounds
  if Torque>T_max then Torque:=T_max
  else if Torque<T_min then Torque:=T_min;

  // To avoid funny stuff in the beginning, damp the beginning of the output
  // aero forces are only applied later, so there shouldn't be any aero T yet
  if stepno < 5 then Torque:=0;

  // for the DLL control, mbdy moment_ext is in Nm, so no scaling required
  outputsignals[0]:=Torque ;
  outputsignals[1]:=K ;

  //writeln(stepno, K, rpm, rpm_filt, outputsignals[0]);
end;

exports init,update;
begin
// Main body
end.




