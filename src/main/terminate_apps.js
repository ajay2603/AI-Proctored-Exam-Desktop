import { exec } from 'child_process'
import { app, ipcMain } from 'electron'

export async function terminateAppProcessesWin() {
  const currentAppPID = process.pid
  const currentAppName = app.getName()
  console.log(`Current app process name: ${currentAppName}, PID: ${currentAppPID}`)

  // Wrap the PowerShell command in quotes and use -Command parameter
  const psCommand =
    'powershell.exe -Command "Get-Process | Where-Object {$_.MainWindowTitle} | Select-Object Name, Id, MainWindowTitle"'

  return new Promise((resolve, reject) => {
    exec(psCommand, (error, stdout, stderr) => {
      if (error) {
        console.error(`Error executing PowerShell command: ${error.message}`)
        reject(error)
        return
      }
      if (stderr) {
        console.error(`stderr: ${stderr}`)
        reject(new Error(stderr))
        return
      }

      try {
        // Skip header lines and empty lines
        const processesList = stdout
          .split('\n')
          .slice(3) // Skip the header lines
          .filter((line) => line.trim() !== '') // Remove empty lines
          .map((line) => {
            const [name, id, ...titleParts] = line.trim().split(/\s+/)
            return {
              name: name || 'Unknown',
              pid: parseInt(id) || 0,
              title: titleParts.join(' ') || 'No Title'
            }
          })
          .filter(
            (proc) =>
              proc.pid !== currentAppPID && // Skip current app process
              proc.name.toLowerCase() !== 'code' &&
              proc.name.toLowerCase() !== 'taskmgr' &&
              proc.name.toLowerCase() !== 'systemsettings' &&
              proc.name.toLowerCase() !== 'textinputhost' // Skip code.exe (Visual Studio Code)
          )

        console.log('Found processes:', processesList)

        // Terminate processes
        processesList.forEach((process) => {
          console.log(
            `Terminating process: ${process.name} (PID: ${process.pid}, Title: ${process.title})`
          )

          exec(`taskkill /PID ${process.pid} /F`, (error, stdout, stderr) => {
            if (error) {
              console.error(
                `Error terminating ${process.name} (PID: ${process.pid}): ${error.message}`
              )
              return
            }
            if (stderr) {
              console.error(
                `stderr while terminating ${process.name} (PID: ${process.pid}): ${stderr}`
              )
              return
            }
            console.log(`Successfully terminated ${process.name} (PID: ${process.pid})`)
          })
        })

        resolve(processesList) // Resolve with the list of terminated processes
      } catch (parseError) {
        console.error('Error parsing process list:', parseError)
        reject(parseError)
      }
    })
  })
}

var terminationId

app.whenReady().then(() => {
  console.log('redy in term')

  ipcMain.on('on-renderer', (_, data) => {
    console.log('in term')
    console.log(data)
    if (data.event !== 'pre-exam') {
      return
    }

    if (terminationId) {
      clearInterval(terminationId)
      terminationId = null
    }

    terminationId = setInterval(() => {
      terminateAppProcessesWin()
        .then((_) => {})
        .catch((err) => {
          console.log(err)
        })
    }, 2000)
  })

  ipcMain.on('post-exam', () => {
    if (terminationId) {
      clearInterval(terminationId)
      terminationId = null
    }
  })
})
