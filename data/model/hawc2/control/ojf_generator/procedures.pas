unit procedures;
interface
  uses global,math;

Function interpolate(x,x0,x1,f0,f1:single):single;

function firstorderfilter(deltat:single;
                          stepno:integer;
                          var firstordervar:Tfirstordervar;
                          x:single):single;

function ave_filter(deltat:single;
                    stepno:integer;
                    var avefiltervar:Tavefiltervar;
                    x:single):single;

Function GetTref(ogagen:single;
                 pit_com:single;
                 oga_ref:single;
                 pit_limit:single;
                 stepno:integer):single;

function butterworth_h2(stepno:integer;
                      var buttervar:Tbuttervar;
                      input:single):single;

function butterworth_orig(stepno:integer;
                      var buttervar:Tbuttervar;
                      input:single):single;

function PID(stepno:integer;
              deltat :single;
              kgain  :single;
              var PIDvar:Tpidvar;
              error : single):single;

function lowpass2orderfilt(deltat: single;
                           stepno: integer;
                           var filt_var: Tlowpass2order;
                           input : single):single;

Function GetOptiPitch(wsp:single):single;

implementation

Function interpolate(x,x0,x1,f0,f1:single):single;
begin
  if x0=x1 then interpolate:=f0
  else interpolate:=(x-x1)/(x0-x1)*f0+(x-x0)/(x1-x0)*f1;
end;

Function GetTref(ogagen:single;
                 pit_com:single;
                 oga_ref:single;
                 pit_limit:single;
                 stepno:integer):single;
var
  x,x0,x1,f0,f1,Tref : single;
  i : integer;
begin  
  if pit_com>pit_limit then ogagen:=oga_ref;
  with wpdatavar do
  begin
    i:=0;
    try
    repeat
      inc(i);
    until (wpdata[i,1]>ogagen) or (i=lines);
    except
      writeln('error 1a getTref');
      writeln('ogagen',ogagen);
      writeln('i',i);
      writeln('lines ' ,lines);
    end;
    if (i=lines) then
      Tref:=wpdata[i,2]
    else begin
      if i=1 then i:=2;

      x:=ogagen;
      x0:=wpdata[i-1,1];
      x1:=wpdata[i,1];
      f0:=wpdata[i-1,2];
      f1:=wpdata[i,2];
      Tref:=interpolate(x,x0,x1,f0,f1);
      if pit_com>pit_limit then Tref:=regulatorvar.T_max;
    end;
    GetTref:=Tref;
  end;
end;

function firstorderfilter(deltat:single;
                          stepno:integer;
                          var firstordervar:Tfirstordervar;
                          x:single):single;
var
  y :single;
begin
  with firstordervar do
  begin
    if stepno=1 then
    begin
      x1_old:=x;
      y1_old:=x;
      y:=x;
    end
    else
    begin
      if stepno>stepno1 then
      begin
        x1_old:=x1;
        y1_old:=y1
      end;
      y:=(deltat*(x+x1_old-y1_old)+2*tau*y1_old)/(deltat+2*tau);
    end;
    x1:=x;
    y1:=y;
    stepno1:=stepno
  end;
  firstorderfilter:=y;
end;

function butterworth_h2(stepno:integer;
                      var buttervar:Tbuttervar;
                      input:single):single;
var
  i:integer;
  temp1,temp2:single;
begin
   with buttervar do
   begin
      IF (STEPNO=1) THEN
        for i:=1 to 3 do
        begin
          inp_old[i]:=input;
          tem_old[i]:=input;
          oup_old[i]:=input;
        end
      else
      begin
        if stepno>stepno1 then
        begin
          inp[3]:=inp[2];
          inp[2]:=inp[1];
          inp[1]:=input;
          inp_old:=inp;
          tem_old:=tem;
          oup_old:=oup;
        end
        else
        begin
          inp[1]:=input;
          inp_old[1]:=input;
        end;

        TEMP1:=butter_a[1,1]*inp_old[1]+butter_a[1,2]*inp_old[2]+butter_a[1,3]*inp_old[3]
              -butter_b[1,1]*tem_old[1]-butter_b[1,2]*tem_old[2];

        tem[3]:=tem_old[2];
        tem[2]:=tem_old[1];
        tem[1]:=temp1;

        TEMP2:=butter_a[2,1]*tem[1]+butter_a[2,2]*tem[2]+butter_a[2,3]*tem[3]
            -butter_b[2,1]*oup_old[1]-butter_b[2,2]*oup_old[2];

        oup[3]:=oup_old[2];
        oup[2]:=oup_old[1];
        oup[1]:=temp2;
      end;
      butterworth_h2:=oup[1];
      stepno1:=stepno;
   end;
end;

function butterworth_orig(stepno:integer;
                      var buttervar:Tbuttervar;
                      input:single):single;
var
  i:integer;
  temp:single;
begin
   with buttervar do
   begin
      IF (STEPNO=1) THEN
        for i:=1 to 3 do
        begin
          inp[i]:=input;
          tem[i]:=input;
          oup[i]:=input;
        end;

      inp[3]:=inp[2];
      inp[2]:=inp[1];
      inp[1]:=input;

      TEMP:=butter_a[1,1]*inp[1]+butter_a[1,2]*inp[2]+butter_a[1,3]*inp[3]
            -butter_b[1,1]*tem[1]-butter_b[1,2]*tem[2];

      tem[3]:=tem[2];
      tem[2]:=tem[1];
      tem[1]:=temp;

      TEMP:=butter_a[2,1]*tem[1]+butter_a[2,2]*tem[2]+butter_a[2,3]*tem[3]
          -butter_b[2,1]*oup[1]-butter_b[2,2]*oup[2];

      oup[3]:=oup[2];
      oup[2]:=oup[1];
      oup[1]:=temp;

      butterworth_orig:=oup[1];
   end;
end;

function PID(stepno:integer;
              deltat :single;
              kgain  :single;
              var PIDvar:Tpidvar;
              error : single):single;
const
  eps = 1.0e-6 ;              
begin
  with PIDvar do
  begin
    if stepno=1 then
    begin
      outset1:=0;
      outres1:=0;
      error1:=0;
      error1_old:=0.0;
      outset1_old:=0.0;
      outres1_old:=0.0;
    end;
    if stepno>stepno1 then
    begin
      outset1_old:=outset1;
      outres1_old:=outres1;
      error1_old:=error1;
    end;
    outset:=outset1_old+error*Kgain*Kint*deltat;

    if (outset<outmin) then outset:=outmin
    else if (outset>outmax) then outset:=outmax;
    // check for max velocity
    if velmax>eps then
      if (abs(outset-outset1_old)/deltat)>velmax then outset:=outset1_old+sign(outset-outset1_old)*velmax*deltat;

    outpro:=Kgain*Kpro*error;
    outdif:=Kgain*Kdif*(error-error1_old)/deltat;
    outres:=outset+outpro+outdif;
    if (outres<outmin) then outres:=outmin
    else if (outres>outmax) then outres:=outmax;

    // check for max velocity
    if velmax>eps then
    if (abs(outres-outres1_old)/deltat)>velmax then outres:=outres1_old+sign(outres-outres1_old)*velmax*deltat;

    outset1:=outset;
    outres1:=outres;
    error1:=error;
    stepno1:=stepno;
    if stepno=0 then PID:=0
    else PID:=outres;
  end;
end;


function lowpass2orderfilt(deltat: single;
                           stepno: integer;
                           var filt_var: Tlowpass2order;
                           input : single):single;
var
   A,temp: single;
begin
  with filt_var do
  begin
    if (stepno=1)and(stepno>stepno1) then
    begin
      omf0:=input;
      omf1:=input;
      omf0_old:=omf0;
      omf1_old:=omf1;
      temp:=input;
    end
    else
    begin
      if stepno>stepno1 then
      begin
        omf0_old:=omf0;
        omf1_old:=omf1;
      end;
      A:=f0*2*pi*deltaT;
      temp:=((input-Omf0_old)*A*A+2*Omf0_old+(ksi*A-1)*Omf1_old)/(1+ksi*A);
    end;
    Omf1:=Omf0_old;
    Omf0:=temp;
    stepno1:=stepno;
  end;
  lowpass2orderfilt:=temp;
end;

Function GetOptiPitch(wsp:single):single;
var
  x,x0,x1,f0,f1,pitch : single;
  i : integer;
begin
  with regulatorvar.OPdatavar do
  begin
    i:=0;
    repeat
      inc(i);
    until (wpdata[i,1]>wsp) or (i>lines);
    if i=1 then i:=2;
    x:=wsp;
    x0:=wpdata[i-1,1];
    x1:=wpdata[i,1];
    f0:=wpdata[i-1,2];
    f1:=wpdata[i,2];
    Pitch:=interpolate(x,x0,x1,f0,f1);
    GetOptiPitch:=Pitch;
  end;
end;

function ave_filter(deltat:single;
                    stepno:integer;
                    var avefiltervar:Tavefiltervar;
                    x:single):single;
var
  i :integer;
begin
  with avefiltervar do
  begin
    if (stepno=1) then    // initialize
    begin
      size:=trunc(time/deltat);
      if (length(xvars)<>size) then setlength(xvars,size);
      for i:=0 to size-1 do xvars[i]:=x;
    end
    else
    begin
      if stepno>stepno1 then
      begin
        for i:=size-1 downto 1 do xvars[i]:=xvars[i-1];
      end;
      xvars[0]:=x;
    end;
    stepno1:=stepno;
    ave_filter:=mean(xvars);
  end;
end;

end.


