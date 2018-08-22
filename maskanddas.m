% Implementation of paper 'Phase-Based Dual-Microphone Robust Speech
% Enhancement'
% If you have any problem, contact me by email: destinygxx@gmail.com

clear
clc

%% load .wav

audioname   = 'carmix';
[x, fs0 ]   = audioread([audioname,'.wav']);
fs          = 16e3;
x           = resample(x,fs,fs0);
Nmic        = size(x,2);
if Nmic ~= 2
    error('The number of channels does not equal 2!')
end

%% set parameters

NFFT        = 256;
fragsize    = 256; % fragsize is the length of analysis window
overlap     = 0.5;
GAMMA       = 5;

%% set window

win_1       = hanning(fragsize).';
win_2       = hanning(NFFT).';
win_pre     = sqrt(win_1);
win_post    = sqrt(win_2);
win_pre     = sqrt(win_pre);
win_post    = sqrt(win_post);
scale_fac   = sqrt(NFFT/sum(win_1));
scale_postfac = 1.0 / sqrt(NFFT/sum(win_2));
win_pre     = win_pre * scale_fac;
win_post    = win_post * scale_postfac;

%% cut frames

% ptout = zeros(Nmic, Nframe, fragsize);
for m = 1:Nmic
    [ptout(m,:,:), Nframe] = cutframe(x(:,m), fragsize, overlap);
end

%% process
% Step1: mask; Step2: DAS

amp_old = zeros(Nmic,Nframe,NFFT/2+1);
pha_old = zeros(Nmic,Nframe,NFFT/2+1);
amp_new = zeros(Nmic,Nframe,NFFT/2+1);
pxx     = zeros(Nmic,NFFT);
thetatk = zeros(Nframe,NFFT/2+1);
eta     = zeros(Nframe,NFFT/2+1);
yxx     = zeros(Nframe,NFFT);
yout    = zeros(Nframe,NFFT);
yy      = zeros(1,NFFT);

for n = 1:Nframe
    % apply window and fft
    pxx(1,:) = fft( reshape(ptout(1,n,:),1,fragsize) .* win_pre , NFFT);
    pxx(2,:) = fft( reshape(ptout(2,n,:),1,fragsize) .* win_pre , NFFT);
    
    % get amplitude and phase of TF
    amp_old(1,n,:) = abs(pxx(1,1:NFFT/2+1));
    amp_old(2,n,:) = abs(pxx(2,1:NFFT/2+1));
    pha_old(1,n,:) = angle(pxx(1,1:NFFT/2+1));
    pha_old(2,n,:) = angle(pxx(2,1:NFFT/2+1));

    % get theta(t,k)
    thetatk(n,:) = pha_old(1,n,:) - pha_old(2,n,:);
%     index = find(thetatk(n,:)<-pi);
%     thetatk(n,index) = -pi;
%     index = find(thetatk(n,:)>pi);
%     thetatk(n,index) = pi;
    
    % get scaling factor
    eta(n,:) = 1 ./ (1 + GAMMA * thetatk(n,:).^2);
    
    % apply scaling factor to TF of the two channels
    amp_new(1,n,:) = reshape(amp_old(1,n,:),1,NFFT/2+1) .* eta(n,:);
    amp_new(2,n,:) = reshape(amp_old(2,n,:),1,NFFT/2+1) .* eta(n,:);
    
    % DAS
    yxx(n,1:NFFT/2+1) = mean([reshape(amp_new(1,n,:).*exp(1i*pha_old(1,n,:)),1,NFFT/2+1);...
                     reshape(amp_new(2,n,:).*exp(1i*pha_old(1,n,:)),1,NFFT/2+1)]);
    yxx(n,NFFT/2+2:end) = conj(fliplr(yxx(n,2:NFFT/2)));
    
    % ifft
    yout(n,:) = real(ifft(yxx(n,:).*win_post,NFFT));
end

%% ola

y = ola(yout,fragsize,overlap);

%% write .wav

audiowrite([audioname,'_out.wav'], y, fs);
