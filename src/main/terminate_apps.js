import { dialog } from 'electron'
import { exec } from 'child_process'
import { app } from 'electron' // Import app module from Electron

export async function terminateAppProcessesWin() {
  const psList = (await import('ps-list')).default

  // Get list of all running processes
  const processes = await psList()

  // Log the current app process name and PID
  const currentAppPID = process.pid // PID of the current Electron process
  const currentAppName = app.getName() // Get the app name from Electron's app module
  console.log(`Current app process name: ${currentAppName}, PID: ${currentAppPID}`)

  // Find the PID of explorer.exe
  const explorerProcess = processes.find((process) => process.name === 'explorer.exe')

  if (explorerProcess) {
    const explorerPID = explorerProcess.pid

    // Find all child processes of explorer.exe (those whose parent PID is the explorer PID)
    const childProcesses = processes.filter((process) => process.ppid === explorerPID)

    // Log and terminate the child processes
    if (childProcesses.length > 0) {
      childProcesses.forEach((process) => {
        // Skip the current app process
        if (process.pid === currentAppPID) {
          console.log(`Skipping current app process: ${currentAppName} (PID: ${currentAppPID})`)
          return
        }

        // Skip vscode (code.exe) process
        if (process.name.toLowerCase() === 'code.exe') {
          console.log(`Skipping vscode process: ${process.name} (PID: ${process.pid})`)
          return
        }

        console.log(`Terminating child process of explorer.exe: ${process.name} (PID: ${process.pid})`)

        // Terminate each child process
        exec(`taskkill /PID ${process.pid} /F`, (error, stdout, stderr) => {
          if (error) {
            console.error(
              `Error terminating ${process.name} (PID: ${process.pid}): ${error.message}`
            )

            // If unable to terminate, ask the user to close the process manually
            dialog.showMessageBox({
              type: 'warning',
              title: 'Unable to Terminate Process',
              message: `Could not terminate the process ${process.name} (PID: ${process.pid}). Please close it manually.`,
              buttons: ['OK']
            })
            return
          }
          if (stderr) {
            console.error(`stderr: ${stderr}`)
            return
          }
          console.log(`Successfully terminated ${process.name} (PID: ${process.pid})`)
        })
      })
    } else {
      console.log('No child processes found for explorer.exe.')
    }
  } else {
    console.log('Explorer.exe process not found.')
  }
}
