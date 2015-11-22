library yaw_control;
uses
  SysUtils,
  Classes,
  Dialogs,
  global in 'global.pas',
  procedures in 'procedures.pas';

Type
  array_1000 = array[0..9] of double;
Var
  lowpass2order : Tlowpass2order;
  time_stop     : double;
  time_old      : double;
  stepno        : integer;
  Kp            : double;
  Ki            : double;
  Kd            : double;
  outmax        : double;
  outmin        : double;
  integral      : double;
  err_prev      : double;
{$R *.res}


procedure init(var InputSignals: array_1000;var OutputSignals: array_1000); cdecl;
begin
  time_stop:=Inputsignals[0]  ;
  lowpass2order.f0:=Inputsignals[1]  ;
  lowpass2order.ksi:=Inputsignals[2]  ;
  Kp:=Inputsignals[3]  ;
  Ki:=Inputsignals[4] ;
  Kd:=Inputsignals[5]  ;
  outmin:=Inputsignals[6]  ;
  outmax:=Inputsignals[7]  ;
  outputsignals[0]:=1.0;
  time_old:=0.0;
  stepno:=0;
  integral:=0;
  err_prev:=0;
end;

procedure update(var InputSignals: array_1000;var OutputSignals: array_1000); cdecl;
var
  time : double;
  x    : double;
  ref  : double;
  err  : double;
  deltat : double;
  dif    : double;
begin
  time:=inputsignals[0];
  deltat:=0.0;               // not really necessay (compiler complaint)
  //writeln(time, inputsignals[1], inputsignals[2]);
  if time>time_old then
  begin
    deltat:=time-time_old;
    time_old:=time;
    inc(stepno);
  end;
  if time<time_stop then begin
    x:=inputsignals[1];
    ref:=inputsignals[2];
    //PID filter based on Wikipedia pseudeo code for ideal PID
    //previous_error = setpoint - process_feedback
    //integral = 0
    //start:
    //  wait(dt)
    //  error = setpoint - process_feedback
    //  integral = integral + (error*dt)
    //  derivative = (error - previous_error)/dt
    //  output = (Kp*error) + (Ki*integral) + (Kd*derivative)
    //  previous_error = error
    //  goto start

    err:=lowpass2orderfilt(deltat,stepno,lowpass2order,x)-ref;
    //err:=x-ref;
    integral:=integral + (err*deltat);
    dif:= (err - err_prev)/deltat;
    outputsignals[0]:=(Kp*err) + (Ki*integral) + (Kd*dif);
    err_prev:= err;
    // cut off if outputsignal is higher or lower than the specified boundaries
    if (outputsignals[0]>outmax) then outputsignals[0]:=outmax
    else if (outputsignals[0]<outmin) then outputsignals[0]:=outmin;
    //else writeln((Kp*err),(Ki*integral),(Kd*dif));
  end else
    outputsignals[0]:=0.0 ;
    
  // To avoid funny stuff in the beginning, damp the beginning of the output
  if stepno < 5 then outputsignals[0]:=0;

//  writeln(outputsignals[0]);
end;

exports init,update;
begin
// Main body
end.




