function backgroundProcesses3dBrain(kickStartData)
global g_stopBackgroundProcesses

g_stopBackgroundProcesses = false;
try
    displayQ = kickStartData{1};
catch ME
    fprintf("Display Error: %s\n", ME.identifier, ME.message);
end
try
    validationQ = kickStartData{2};
catch ME
    fprintf("Validation Error: %s\n", ME.identifier, ME.message);
end
maxPollTime = 1;
while ~g_stopBackgroundProcesses && exist('displayQ','var')
    try
        if exist('displayQ','var')
            displayTrainingProgress3dBrain(displayQ,maxPollTime);
        end
    catch ME
        fprintf("Display Error (%s): %s\n", ME.identifier, ME.message);
    end
    try
        if exist('validationQ','var')
            processValidation3dBrain(validationQ,displayQ,maxPollTime);
        end
    catch ME
        fprintf("Validation Error (%s): %s\n", ME.identifier, ME.message);
    end
end
end

