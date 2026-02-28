import prisma from '../prisma';
import { Job } from '@prisma/client';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT, getTrainingFolder, getHFToken, getLunaImageGenPath } from '../paths';
const isWindows = process.platform === 'win32';

const startAndWatchJob = (job: Job) => {
  // starts and watches the job asynchronously
  return new Promise<void>(async (resolve, reject) => {
    const jobID = job.id;

    // setup the training
    const trainingRoot = await getTrainingFolder();

    const trainingFolder = path.join(trainingRoot, job.name);
    if (!fs.existsSync(trainingFolder)) {
      fs.mkdirSync(trainingFolder, { recursive: true });
    }

    // make the config file
    const configPath = path.join(trainingFolder, '.job_config.json');

    //log to path
    const logPath = path.join(trainingFolder, 'log.txt');

    try {
      // if the log path exists, move it to a folder called logs and rename it {num}_log.txt, looking for the highest num
      // if the log path does not exist, create it
      if (fs.existsSync(logPath)) {
        const logsFolder = path.join(trainingFolder, 'logs');
        if (!fs.existsSync(logsFolder)) {
          fs.mkdirSync(logsFolder, { recursive: true });
        }

        let num = 0;
        while (fs.existsSync(path.join(logsFolder, `${num}_log.txt`))) {
          num++;
        }

        fs.renameSync(logPath, path.join(logsFolder, `${num}_log.txt`));
      }
    } catch (e) {
      console.error('Error moving log file:', e);
    }

    // update the config dataset path
    const jobConfig = JSON.parse(job.job_config);
    jobConfig.config.process[0].sqlite_db_path = path.join(TOOLKIT_ROOT, 'aitk_db.db');

    // write the config file
    fs.writeFileSync(configPath, JSON.stringify(jobConfig, null, 2));

    // Determine if this is a Z-Image slider job (LUNA pipeline)
    const jobType = jobConfig?.config?.process?.[0]?.type;
    const isZImageJob = jobType === 'zimage_slider';

    let pythonPath = 'python';
    let runFilePath: string;
    let cwd: string;

    if (isZImageJob) {
      // ---- Z-Image Slider: use LUNA-Image-Gen's Python & bridge script ----
      const lunaPath = await getLunaImageGenPath();
      if (!lunaPath || lunaPath.trim() === '') {
        console.error('LUNA_IMAGEGEN_PATH not configured in Settings');
        await prisma.job.update({
          where: { id: jobID },
          data: {
            status: 'error',
            info: 'Error: LUNA_IMAGEGEN_PATH not set in Settings. Go to Settings and set the path to your LUNA-Image-Gen project.',
          },
        });
        return;
      }

      cwd = lunaPath;
      runFilePath = path.join(lunaPath, 'run_slider_from_ui.py');

      // Use LUNA-Image-Gen's venv
      if (fs.existsSync(path.join(lunaPath, '.venv'))) {
        pythonPath = isWindows
          ? path.join(lunaPath, '.venv', 'Scripts', 'python.exe')
          : path.join(lunaPath, '.venv', 'bin', 'python');
      } else if (fs.existsSync(path.join(lunaPath, 'venv'))) {
        pythonPath = isWindows
          ? path.join(lunaPath, 'venv', 'Scripts', 'python.exe')
          : path.join(lunaPath, 'venv', 'bin', 'python');
      }

      if (!fs.existsSync(runFilePath)) {
        console.error(`run_slider_from_ui.py not found at path: ${runFilePath}`);
        await prisma.job.update({
          where: { id: jobID },
          data: {
            status: 'error',
            info: `Error: run_slider_from_ui.py not found at ${runFilePath}`,
          },
        });
        return;
      }
    } else {
      // ---- Standard ai-toolkit job ----
      cwd = TOOLKIT_ROOT;

      // use .venv or venv if it exists
      if (fs.existsSync(path.join(TOOLKIT_ROOT, '.venv'))) {
        pythonPath = isWindows
          ? path.join(TOOLKIT_ROOT, '.venv', 'Scripts', 'python.exe')
          : path.join(TOOLKIT_ROOT, '.venv', 'bin', 'python');
      } else if (fs.existsSync(path.join(TOOLKIT_ROOT, 'venv'))) {
        pythonPath = isWindows
          ? path.join(TOOLKIT_ROOT, 'venv', 'Scripts', 'python.exe')
          : path.join(TOOLKIT_ROOT, 'venv', 'bin', 'python');
      }

      runFilePath = path.join(TOOLKIT_ROOT, 'run.py');
      if (!fs.existsSync(runFilePath)) {
        console.error(`run.py not found at path: ${runFilePath}`);
        await prisma.job.update({
          where: { id: jobID },
          data: {
            status: 'error',
            info: `Error launching job: run.py not found`,
          },
        });
        return;
      }
    }

    const additionalEnv: any = {
      AITK_JOB_ID: jobID,
      CUDA_DEVICE_ORDER: 'PCI_BUS_ID',
      CUDA_VISIBLE_DEVICES: `${job.gpu_ids}`,
      IS_AI_TOOLKIT_UI: '1',
    };

    // HF_TOKEN
    const hfToken = await getHFToken();
    if (hfToken && hfToken.trim() !== '') {
      additionalEnv.HF_TOKEN = hfToken;
    }

    // Add the --log argument to the command
    const args = [runFilePath, configPath, '--log', logPath];

    try {
      let subprocess;

      if (isWindows) {
        // Spawn Python directly on Windows so the process can survive parent exit
        subprocess = spawn(pythonPath, args, {
          env: {
            ...process.env,
            ...additionalEnv,
          },
          cwd: cwd,
          detached: true,
          windowsHide: true,
          stdio: 'ignore', // don't tie stdio to parent
        });
      } else {
        // For non-Windows platforms, fully detach and ignore stdio so it survives daemon-like
        subprocess = spawn(pythonPath, args, {
          detached: true,
          stdio: 'ignore',
          env: {
            ...process.env,
            ...additionalEnv,
          },
          cwd: cwd,
        });
      }

      // Important: let the child run independently of this Node process.
      if (subprocess.unref) {
        subprocess.unref();
      }

      // Optionally write a pid file for future management (stop/inspect) without keeping streams open
      try {
        fs.writeFileSync(path.join(trainingFolder, 'pid.txt'), String(subprocess.pid ?? ''), { flag: 'w' });
      } catch (e) {
        console.error('Error writing pid file:', e);
      }

      // (No stdout/stderr listeners — logging should go to --log handled by your Python)
      // (No monitoring loop — the whole point is to let it live past this worker)
    } catch (error: any) {
      // Handle any exceptions during process launch
      console.error('Error launching process:', error);

      await prisma.job.update({
        where: { id: jobID },
        data: {
          status: 'error',
          info: `Error launching job: ${error?.message || 'Unknown error'}`,
        },
      });
      return;
    }
    // Resolve the promise immediately after starting the process
    resolve();
  });
};

export default async function startJob(jobID: string) {
  const job: Job | null = await prisma.job.findUnique({
    where: { id: jobID },
  });
  if (!job) {
    console.error(`Job with ID ${jobID} not found`);
    return;
  }
  // update job status to 'running', this will run sync so we don't start multiple jobs.
  await prisma.job.update({
    where: { id: jobID },
    data: {
      status: 'running',
      stop: false,
      info: 'Starting job...',
    },
  });
  // start and watch the job asynchronously so the cron can continue
  startAndWatchJob(job);
}
