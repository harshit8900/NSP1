function face_emotion_recognition()
    % Create a figure window
    fig = uifigure('Name', 'Real-Time Emotion Recognition', 'Position', [500 300 400 250]);
    
    % Create Start Button
    startBtn = uibutton(fig, 'push', 'Text', 'Start Detection', ...
        'Position', [50 150 120 50], ...
        'ButtonPushedFcn', @(btn,event) startDetection(fig));
    
    % Create Stop Button
    stopBtn = uibutton(fig, 'push', 'Text', 'Stop Detection', ...
        'Position', [230 150 120 50], ...
        'ButtonPushedFcn', @(btn,event) stopDetection(fig));
    
    % Create Label for FPS
    fpsLabel = uilabel(fig, ...
        'Position', [150 50 100 30], ...
        'Text', 'FPS: --', ...
        'FontSize', 16);
    
    % Store app data (shared variables)
    setappdata(fig, 'running', false);
    setappdata(fig, 'fpsLabel', fpsLabel);
end

function startDetection(fig)
    % Check if already running
    if getappdata(fig, 'running')
        return;
    end
    setappdata(fig, 'running', true);
    
    % Load Pre-trained Model
    load('trainedEmotionNet.mat','net');
    
    % Set up webcam
    cam = webcam;
    
    % Create face detector
    faceDetector = vision.CascadeObjectDetector();
    
    % Start video player
    videoPlayer = vision.DeployableVideoPlayer();
    
    fpsLabel = getappdata(fig, 'fpsLabel');
    tic; % Start timer for FPS calculation
    frameCount = 0;
    
    while isOpen(videoPlayer) && getappdata(fig, 'running')
        frameCount = frameCount + 1;
        
        % Capture frame
        img = snapshot(cam);
        
        % Detect faces
        bbox = step(faceDetector, img);
        
        for i = 1:size(bbox,1)
            face = imcrop(img, bbox(i,:));
            face = imresize(face, [48 48]);
            grayFace = rgb2gray(face);
            grayFace = im2single(grayFace);
            grayFace = reshape(grayFace, [48,48,1]);
            
            % Predict emotion
            label = classify(net, grayFace);
            
            % Annotate
            img = insertObjectAnnotation(img, 'rectangle', bbox(i,:), char(label), 'TextBoxOpacity', 0.8, 'FontSize', 14);
        end
        
        % Display
        step(videoPlayer, img);
        
        % Update FPS every 10 frames
        if mod(frameCount,10) == 0
            elapsedTime = toc;
            fps = frameCount / elapsedTime;
            fpsLabel.Text = sprintf('FPS: %.2f', fps);
        end
    end
    
    % Cleanup
    if isvalid(cam)
        clear cam;
    end
    if isvalid(videoPlayer)
        release(videoPlayer);
    end
end

function stopDetection(fig)
    setappdata(fig, 'running', false);
end
