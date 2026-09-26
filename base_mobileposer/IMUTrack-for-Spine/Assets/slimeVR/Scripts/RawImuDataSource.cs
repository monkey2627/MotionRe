using System;
using System.Collections.Generic;

namespace SpineFlow.RawImu
{
    /// <summary>
    /// Selects direct physical-tracker data. SolarXR is preferred because the
    /// active trackers are connected to SlimeVR Server; WitMotion stays as a
    /// compatibility fallback for older hardware.
    /// </summary>
    public static class RawImuDataSource
    {
        public static bool TryStart(out string error)
        {
            return SlimeVrRawImuDataSource.TryStart(out error);
        }

        public static bool TryCopyLatestSamples(List<RawImuSensorSample> destination,
            out string error)
        {
            if (destination == null) throw new ArgumentNullException(nameof(destination));

            SlimeVrRawImuDataSource.TryStart(out string startError);
            if (SlimeVrRawImuDataSource.TryCopyLatestSamples(destination, out string solarError) &&
                destination.Count > 0)
            {
                error = null;
                return true;
            }

            if (WitRawImuDataSource.TryCopyLatestSamples(destination, out string witError) &&
                destination.Count > 0)
            {
                error = null;
                return true;
            }

            destination.Clear();
            error = FirstMessage(solarError, startError, witError,
                "Waiting for direct SlimeVR tracker data on ws://127.0.0.1:21110 ...");
            return false;
        }

        private static string FirstMessage(params string[] messages)
        {
            foreach (string message in messages)
                if (!string.IsNullOrWhiteSpace(message)) return message;
            return null;
        }
    }
}
