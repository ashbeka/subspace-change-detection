% %%%% 作業フォルダに移動
% cd('D:\Users\nosaka\Documents\MATLAB\');

% %%%% 自分の作ったライブラリを全部追加（指定したディレクトリ以下を全部パスに追加）
%  addpath(genpath('C:\Users\nosaka\Documents\My Dropbox\cvtToolBox'));
% addpath('D:\Users\nosaka\Documents\My Dropbox\cvtToolBox');
 
%% diary コマンドが作るファイルに拡張子を追加
set(0, 'DiaryFile', 'diary.txt');


%%% % Figureウインドウ関係のデフォルト設定 -- 以下は参考程度に

scrsz = get(0, 'ScreenSize');    % スクリーンサイズの取得

% Figureウインドウの位置とサイズのデフォルト
set(0, 'defaultFigurePosition', [200 scrsz(4)-620 480 360]);

% Figure の紙の大きさ
set(0, 'defaultFigurePaperType', 'A4');

% Figureの寸法の単位
set(0, 'defaultFigurePaperUnits', 'inches');

% 黒背景のものprintするときに色を変更するか？
set(0, 'defaultFigureInvertHardcopy', 'on');

% 軸のボックスをonに
set(0, 'defaultAxesBox', 'on');

%% 色関係
% テキストカラー
set(0, 'defaultTextColor', [0 0 0]);

% 各軸の色 (軸の目盛りのテキストも変更）
set(0, 'defaultAxesXColor', [0 0 0]);
set(0, 'defaultAxesYColor', [0 0 0]);
set(0, 'defaultAxesZColor', [0 0 0]);

% パッチのエッジの色
set(0, 'defaultPatchEdgeColor', [0 0 0]);

% Surfaceのエッジの色
set(0, 'defaultSurfaceEdgeColor', [0 0 0]);

% 注釈オブジェクトの線の色　plot の色ではなくannotationの色　Figureウインドウで書ける線の色
set(0, 'defaultLineColor', [1 0 0]);

% Figure ウインドウの色
%    この例では透明にしていますが、[1 1 1]なら白などカラーベクトルで指定できます。
%set(0, 'defaultFigureColor', 'none');
set(0, 'defaultFigureColor', [1 1 1]);


% 軸の色（プロットエリアの色）
set(0, 'defaultAxesColor', [1 1 1]);

% plotなどで自動的に付けられる色の指定
corder = [[1 0 0];[0 0.5000 0];[0 0 1];[1 0.7500 0.7500];[0.7500 0 0.7500];[0.7500 0.7500 0];[0.2500 0.2500 0.2500]];
set(0, 'defaultAxesColorOrder', corder);

% デフォルトのカラーマップ
cmap = jet;
set(0, 'defaultFigureColormap', cmap);

%% フォント関係
% GUIのフォント
set(0, 'defaultUicontrolFontName', 'MS UI Gothic');
% 軸のフォント
set(0, 'defaultAxesFontName', 'Arial');
% タイトル、注釈などのフォント
set(0, 'defaultTextFontName', 'Times');

% GUI のフォントサイズ
set(0, 'defaultUicontrolFontSize', 9);

% 軸のフォントサイズ
set(0, 'defaultAxesFontSize', 12);

% タイトル、注釈などのフォントサイズ
set(0, 'defaultTextFontSize', 12);


%% mail
% SMTPサーバの設定
setpref( 'Internet', 'SMTP_Server', 'smtp.gmail.com');
% SMTPサーバでのユーザ名（必要なら）
setpref( 'Internet', 'SMTP_Username', 'nosyan');
% SMTPサーバでのパスワード（必要なら）
setpref( 'Internet', 'SMTP_Password', 'ninten');
% 自分のアドレス（Fromになる）
setpref('Internet', 'E_mail', 'nosyan+matlab@gmail.com');
props = java.lang.System.getProperties;
props.setProperty('mail.smtp.auth','true');
props.setProperty('mail.smtp.socketFactory.class', 'javax.net.ssl.SSLSocketFactory');
props.setProperty('mail.smtp.socketFactory.port','465');


%% そのほか
% 軸の線の太さ
set(0, 'DefaultAxesLineWidth', 2);

% 注釈オブジェクトの線の太さ
set(0, 'DefaultLineLineWidth', 2);

close all;
clear;

