const fs = require('fs');
const path = require('path');
const { Client } = require('@gradio/client');
const ffmpeg = require('fluent-ffmpeg');
const pMap = require('p-map').default;
const ttsServer = require('./tts_server.json');
const ttsModels = require('./tts_models.json');

// const TTS_SERVER_SELECTED = process.env.TTS_SERVER_SELECTED || 'local'; // Default to 'local' if not set
const TTS_SERVER_SELECTED = process.env.TTS_SERVER_SELECTED || 'onPremise'; // Default to 'onPremise' if not set
// const TTS_MODEL_SELECTED = process.env.TTS_MODEL_SELECTED || 'macos'; // Default to 'macos' if not set
const TTS_MODEL_SELECTED = process.env.TTS_MODEL_SELECTED || 'cpu'; // Default to 'macos' if not set

const TTS_SERVER = ttsServer[TTS_SERVER_SELECTED]; // Use the selected server URL from config
const TTS_MODEL = ttsModels[TTS_MODEL_SELECTED]; // Use the selected model from config

// Function to split text into chapters based on "Chương"
function splitTextIntoChunks(text) {
  const chunks = [];
  const chapterInfo = [];
  const chapterRegex = /Chương \d+/g;
  const matches = [...text.matchAll(chapterRegex)];
  
  if (matches.length === 0) {
    // If no chapters found, return the entire text as one chunk
    return { chunks: [text], chapterInfo: [{ number: 1, title: '' }] };
  }
  
  for (let i = 0; i < matches.length; i++) {
    const startIndex = matches[i].index;
    const endIndex = i < matches.length - 1 ? matches[i + 1].index : text.length;
    const chunk = text.slice(startIndex, endIndex).trim();
    
    if (chunk) {
      chunks.push(chunk);
      
      // Extract chapter number and title
      // Pattern: "Chương 11: Bạch Tinh Xà" or "Chương 11 - Bạch Tinh Xà"
      const firstLine = chunk.split('\n')[0];
      const chapterMatch = firstLine.match(/Chương\s+(\d+)[:\-\s]+(.+)/);
      
      if (chapterMatch) {
        chapterInfo.push({
          number: parseInt(chapterMatch[1]),
          title: chapterMatch[2].trim()
        });
      } else {
        // Fallback: just extract the number
        const numMatch = firstLine.match(/Chương\s+(\d+)/);
        chapterInfo.push({
          number: numMatch ? parseInt(numMatch[1]) : i + 1,
          title: ''
        });
      }
    }
  }
  
  return { chunks, chapterInfo };
}

/**
 * Load the TTS model with progress tracking
 * Server will check if model is already loaded and skip reload if same config
 */
async function loadModel(client) {
  console.log("📦 Loading VieNeu-TTS model...");
  
  try {
    // Use direct array format (not wrapped in data object)
    const result = await client.predict("/load_model", TTS_MODEL);
    
    const statusMsg = result.data[0];
    console.log("Status:", statusMsg);
    
    // Check for already loaded message
    if (statusMsg.includes("đã được tải sẵn") || statusMsg.includes("already")) {
      console.log("⚡ Model was already loaded - no reload needed!");
      return true;
    }
    
    if (statusMsg.includes("❌") || statusMsg.includes("Lỗi")) {
      console.error("❌ Model failed to load:", statusMsg);
      return false;
    }
    
    console.log("✅ Model loaded successfully!");
    return true;
  } catch (error) {
    console.error("❌ Error loading model:", error);
    return false;
  }
}

/**
 * Synthesize speech with progress tracking
 */
async function synthesizeSpeechWithProgress(client, chunk, voiceName, chunkIndex, totalChunks) {
  console.log(`\n[${chunkIndex + 1}/${totalChunks}] Processing chapter...`);
  
  try {
    // Submit the job to track progress
    const job = client.submit("/synthesize_speech", [
      chunk,                        // text
      voiceName,                    // voice_choice
      null,                         // custom_audio
      "",                           // custom_text
      "preset_mode",                // mode_tab (current_mode_state)
      "Standard (Một lần)",         // generation_mode
      true,                         // use_batch
      128,                           // max_batch_size_run
      1.0,                          // temperature
      512                           // max_chars_chunk
    ]);

    let lastProgressMsg = "";
    let isComplete = false;
    let finalAudioPath = null;

    // Track progress - use a try-catch to handle iterator completion properly
    try {
      for await (const message of job) {
        if (message.type === "data" && message.data) {
          // Save the audio path if available
          if (message.data[0]) {
            finalAudioPath = message.data[0];
          }
          
          // Check progress messages
          if (message.data[1]) {
            const progressMsg = message.data[1];
            if (progressMsg && typeof progressMsg === 'string' && progressMsg !== lastProgressMsg) {
              // Only show new progress messages
              console.log(`   📊 ${progressMsg}`);
              lastProgressMsg = progressMsg;
              
              // Check if job is complete
              if (progressMsg.includes('✅ Hoàn tất!') || 
                  progressMsg.includes('File đã lưu tại:')) {
                isComplete = true;
                // Don't break yet, wait a bit for final data
                await new Promise(resolve => setTimeout(resolve, 500));
                break;
              }
            }
          }
        }
      }
    } catch (iterError) {
      // Iterator might throw when complete, which is normal
      if (!iterError.message?.includes('complete')) {
        console.warn(`   ⚠️  Iterator warning:`, iterError.message);
      }
    }

    // If we detected completion and have audio path, return it
    if (isComplete && finalAudioPath) {
      console.log(`   ✅ Complete`);
      return finalAudioPath;
    }

    // Otherwise get the result from job
    const result = await job;
    
    if (result.data && result.data[0]) {
      const audioPath = result.data[0];
      const statusMsg = result.data[1];
      
      // Extract just the success message
      const cleanStatus = statusMsg ? statusMsg.split('\n')[0] : 'Complete';
      console.log(`   ✅ ${cleanStatus}`);
      
      return audioPath;
    } else {
      console.error(`   ❌ No audio data returned`);
      return null;
    }

  } catch (error) {
    console.error(`   ❌ Error:`, error.message || error);
    throw error;
  }
}

async function convertTextToSpeech() {
  // Get input file paths from command line arguments
  const param1 = process.argv[2]; // The folder name (param-1)
  const startChapter = process.argv[3]; // Start chapter number
  const endChapter = process.argv[4]; // End chapter number

  if (!param1 || !startChapter || !endChapter) {
    console.error('Please provide folder name, start chapter, and end chapter as parameters.');
    console.error('Usage: node 12_tts.js <folder> <start> <end>');
    console.error('Example: node 12_tts.js game 1 50');
    process.exit(1); // Exit if parameters are missing
  }

  const param2 = `chuong_${startChapter}-${endChapter}`;
  const textFilePath = path.join(__dirname, `../input/${param1}/chat-gpt/${param2}.txt`);
  if (!fs.existsSync(textFilePath)) {
    console.error(`The file ${textFilePath} does not exist.`);
    process.exit(1); // Exit if the file doesn't exist
  }

  const text = fs.readFileSync(textFilePath, 'utf8');

  // Split the text into chapters based on "Chương"
  const { chunks, chapterInfo } = splitTextIntoChunks(text);

  try {
    // Create the output folder if it doesn't exist
    const outputFolder = path.join(__dirname, `../output/${param1}/mp3/${param2}`);
    if (!fs.existsSync(outputFolder)) {
      fs.mkdirSync(outputFolder, { recursive: true }); // Create folder and any necessary subdirectories
    }

    // Connect to the Gradio client
    console.log("📡 Connecting to VieNeu-TTS API...");
    const client = await Client.connect(TTS_SERVER);
    console.log("✅ Connected!");

    // Load the model - server will skip if already loaded
    const modelLoaded = await loadModel(client);
    
    if (!modelLoaded) {
      console.error("❌ Failed to load model. Exiting.");
      process.exit(1);
    }
    
    // No need to wait if model was already loaded
    // The server response will indicate if it's a fresh load or reuse

    // Process each chunk with progress tracking
    const outputFiles = [];
    
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      
      try {
        const audioPath = await synthesizeSpeechWithProgress(
          client, 
          chunk, 
          "Tuyen", 
          i, 
          chunks.length
        );

        if (audioPath) {
          // Handle Gradio file object or string path
          let downloadUrl;
          let cleanPath;
          
          if (typeof audioPath === 'object' && audioPath !== null) {
            // It's a Gradio file object - use the URL field directly
            downloadUrl = audioPath.url;
            cleanPath = audioPath.path || audioPath.orig_name;
            console.log(`   📥 File object received: ${audioPath.orig_name}`);
          } else if (typeof audioPath === 'string') {
            cleanPath = audioPath;
            // Remove /file= prefix if present
            if (cleanPath.startsWith('/file=')) {
              cleanPath = cleanPath.substring(6);
            }
            downloadUrl = `${TTS_SERVER}/file=${cleanPath}`;
          } else {
            throw new Error(`Unexpected audioPath type: ${typeof audioPath}`);
          }
          
          console.log(`   🔗 Download URL: ${downloadUrl}`);
          
          try {
            const audioResponse = await fetch(downloadUrl);
            
            if (!audioResponse.ok) {
              throw new Error(`HTTP ${audioResponse.status}: ${audioResponse.statusText}`);
            }
            
            const arrayBuffer = await audioResponse.arrayBuffer();
            
            if (arrayBuffer.byteLength === 0) {
              throw new Error('Downloaded file is empty (0 bytes)');
            }
            
            const outputPath = path.join(outputFolder, `${i}.wav`);
            fs.writeFileSync(outputPath, Buffer.from(arrayBuffer));
            
            // Verify the file was written correctly
            const stats = fs.statSync(outputPath);
            if (stats.size === 0) {
              throw new Error('Saved file is empty (0 bytes)');
            }
            
            outputFiles.push(outputPath);
            console.log(`   💾 Saved to: ${outputPath} (${(stats.size / 1024).toFixed(2)} KB)`);
          } catch (downloadError) {
            console.error(`   ❌ Download failed:`, downloadError.message);
            
            // Fallback: Try to copy from output_audio folder if it exists
            const outputAudioPattern = /output_audio\/tts_output_\d+_\d+\.wav/;
            const match = cleanPath.match(outputAudioPattern);
            
            if (match || cleanPath.includes('output_audio/')) {
              const filename = path.basename(cleanPath);
              const localPath = path.join('/Users/danielpham/sync-workspace/05_Stories/VieNeu-TTS/output_audio', filename);
              console.log(`   🔄 Trying local copy from: ${localPath}`);
              
              if (fs.existsSync(localPath)) {
                const outputPath = path.join(outputFolder, `${i}.wav`);
                fs.copyFileSync(localPath, outputPath);
                
                const stats = fs.statSync(outputPath);
                if (stats.size === 0) {
                  throw new Error('Copied file is empty (0 bytes)');
                }
                
                console.log(`   💾 Copied to: ${outputPath} (${(stats.size / 1024).toFixed(2)} KB)`);
                outputFiles.push(outputPath);
              } else {
                throw new Error(`Local file not found: ${localPath}`);
              }
            } else {
              throw downloadError;
            }
          }
        }
      } catch (error) {
        console.error(`\n❌ Failed to process chapter ${i + 1}:`, error);
        if (process.env.STOP_ON_ERROR !== 'false') {
          throw error;
        }
      }
    }

    if (outputFiles.length === 0) {
      console.error('\n❌ No audio files were generated. Exiting.');
      process.exit(1);
    }

    console.log(`\n📦 Successfully generated ${outputFiles.length}/${chunks.length} audio files`);

    // Combine the audio files using ffmpeg
    await combineAudioFiles(outputFiles, outputFolder, param1, param2, chapterInfo);
    
    console.log('\n✅ All processing complete!');
    
    // Close the client connection and exit
    try {
      await client.close();
      console.log('🔌 Client connection closed');
    } catch (closeError) {
      console.warn('⚠️  Warning: Could not close client connection:', closeError.message);
    }
    
    // Force exit after a brief delay to ensure all async operations complete
    setTimeout(() => {
      process.exit(0);
    }, 1000);
    
  } catch (error) {
    console.error('\n❌ Fatal error:', error);
    process.exit(1);
  }
}

// Function to generate chapter menu with timestamps
async function generateChapterMenu(files, param1, param2, chapterInfo) {
  const menuFolder = path.join(__dirname, `../output/${param1}/subtitle`);
  if (!fs.existsSync(menuFolder)) {
    fs.mkdirSync(menuFolder, { recursive: true });
  }

  const menuPath = path.join(menuFolder, `${param2}_menu.txt`);
  const menuLines = [];
  
  console.log('\n📝 Generating chapter menu...');
  
  // Get durations of each file using ffprobe
  const ffprobePromises = files.map(file => {
    return new Promise((resolve, reject) => {
      ffmpeg.ffprobe(file, (err, metadata) => {
        if (err) {
          reject(err);
        } else {
          resolve({
            file,
            duration: metadata.format.duration
          });
        }
      });
    });
  });

  try {
    const fileInfos = await Promise.all(ffprobePromises);
    let currentTime = 0;
    
    fileInfos.forEach((info, index) => {
      // Adjust duration for tempo change (1.1x speed means shorter duration)
      const adjustedDuration = info.duration / 1.1;
      
      // Format timestamp as HH:MM:SS
      const hours = Math.floor(currentTime / 3600);
      const minutes = Math.floor((currentTime % 3600) / 60);
      const seconds = Math.floor(currentTime % 60);
      const timestamp = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
      
      // Get chapter info
      const chapter = chapterInfo[index] || { number: index + 1, title: '' };
      const chapterText = chapter.title 
        ? `Chương ${chapter.number} - ${chapter.title}`
        : `Chương ${chapter.number}`;
      
      menuLines.push(`${timestamp} - ${chapterText}`);
      
      currentTime += adjustedDuration;
    });
    
    // Write menu file
    fs.writeFileSync(menuPath, menuLines.join('\n'));
    console.log(`   ✅ Chapter menu saved to: ${menuPath}`);
    console.log(`   📊 Total chapters: ${files.length}`);
    console.log(`   ⏱️  Total duration: ${Math.floor(currentTime / 60)}m ${Math.floor(currentTime % 60)}s`);
  } catch (err) {
    console.error('   ❌ Error generating chapter menu:', err);
    throw err;
  }
}

// Function to combine audio files into a single MP3 and delete the individual files
async function combineAudioFiles(files, outputFolder, param1, param2, chapterInfo) {
  // Sắp xếp các tệp theo thứ tự tăng dần dựa trên số thứ tự trong tên tệp
  files.sort((a, b) => {
    const getNumber = (file) => {
      const match = file.match(/(\d+)\.wav$/);
      return match ? parseInt(match[1], 10) : 0;
    };
    return getNumber(a) - getNumber(b);
  });

  console.log('\n🔄 Combining audio files with 1.1x speed adjustment...');
  console.log('Danh sách tệp sau khi sắp xếp:', files);

  // Update the combined output path
  const combinedOutputPath = path.join(outputFolder, `${param2}.mp3`);

  // Ensure the target folder exists before writing the combined file
  const targetFolder = path.dirname(combinedOutputPath);
  if (!fs.existsSync(targetFolder)) {
    fs.mkdirSync(targetFolder, { recursive: true });
  }

  // Verify the individual files exist
  files.forEach(file => {
    if (!fs.existsSync(file)) {
      throw new Error(`File not found: ${file}`);
    }
  });

  // Create a temporary file list for ffmpeg in the output folder
  const fileListPath = path.join(outputFolder, 'file-list.txt');
  const fileListContent = files
    .map(file => `file '${path.resolve(file)}'`)
    .join('\n')
    .trim(); // Remove any leading/trailing whitespace

  fs.writeFileSync(fileListPath, fileListContent);

  // Log the file list to debug
  console.log('file-list.txt content:', fs.readFileSync(fileListPath, 'utf8'));

  // Use ffmpeg to combine the audio files
  await new Promise((resolve, reject) => {
    ffmpeg()
      .input(fileListPath)
      .inputOptions('-f', 'concat')
      .inputOptions('-safe', '0')
      .audioFilters('atempo=1.15')
      .audioCodec('libmp3lame')
      .audioBitrate('128k')
      .on('start', (commandLine) => {
        console.log('   🎬 FFmpeg command:', commandLine);
      })
      .on('progress', (progress) => {
        if (progress.percent) {
          process.stdout.write(`\r   ⏳ Progress: ${progress.percent.toFixed(1)}%`);
        }
      })
      .on('end', () => {
        console.log('\r   ✅ FFmpeg processing complete!                    ');
        console.log(`\n✅ Successfully combined audio files into: ${combinedOutputPath}`);
        resolve();
      })
      .on('error', (err) => {
        console.error('\n❌ Error combining audio files:', err);
        reject(err);
      })
      .save(combinedOutputPath);
  });

  // Generate chapter menu with timestamps (before deleting files)
  try {
    await generateChapterMenu(files, param1, param2, chapterInfo);
  } catch (menuError) {
    console.error('   ⚠️  Warning: Failed to generate chapter menu:', menuError);
  }

  // Delete individual chunk files
  console.log('\n🗑️  Cleaning up individual audio files...');
  files.forEach(file => {
    try {
      fs.unlinkSync(file);
      console.log(`   ✅ Deleted: ${path.basename(file)}`);
    } catch (err) {
      console.error(`   ❌ Error deleting ${path.basename(file)}:`, err);
    }
  });

  // Clean up temporary file list
  try {
    fs.unlinkSync(fileListPath);
    console.log('   🗑️  Deleted temporary file list');
  } catch (err) {
    console.error('   ❌ Error deleting file list:', err);
  }
}

convertTextToSpeech();
