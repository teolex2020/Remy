#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif

#ifndef SourceDir
  #define SourceDir "..\..\.build\windows\dist\Remy"
#endif

#ifndef NumericVersion
  #define NumericVersion "0.0.0.0"
#endif

#ifndef OutputDir
  #define OutputDir "..\..\release"
#endif

[Setup]
AppId={{B4856E2D-82C4-4B47-9A9E-08C0A66AFC1D}
AppName=Remy
AppVersion={#AppVersion}
AppPublisher=AuroraSeed
AppPublisherURL=https://github.com/teolex2020/myagent
AppSupportURL=https://github.com/teolex2020/myagent/issues
DefaultDirName={localappdata}\Programs\Remy
DefaultGroupName=Remy
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
OutputDir={#OutputDir}
OutputBaseFilename=RemySetup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
SetupLogging=yes
UninstallDisplayIcon={app}\Remy.exe
CloseApplications=yes
RestartApplications=no
UsePreviousAppDir=yes
VersionInfoVersion={#NumericVersion}
VersionInfoCompany=AuroraSeed
VersionInfoDescription=Remy desktop installer
VersionInfoProductName=Remy
VersionInfoProductVersion={#NumericVersion}

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Remy"; Filename: "{app}\Remy.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Remy"; Filename: "{app}\Remy.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\Remy.exe"; Description: "Launch Remy"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent
