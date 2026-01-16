// Honeypot Configuration Module
const HoneypotManager = (function() {
    
    function initializeEventListeners() {
        const hpSwitch = document.getElementById("honeypot-switch");
        const hpOptions = document.getElementById("honeypot-options");

        if (hpSwitch) {
            hpSwitch.addEventListener("change", function() {
                hpOptions.style.display = this.checked ? "block" : "none";
            });
        }
    }

    function getHoneypotConfig() {
        const isEnabled = document.getElementById("honeypot-switch")?.checked || false;
        
        if (!isEnabled) {
            return null;
        }

        return {
            enabled: true,
            mode: "fixed",
            count: 1, // Enforced constraint: Only 1 Honeypot
            seed: parseFloat(document.getElementById("honeypot-seed").value)
        };
    }

    function setHoneypotConfig(config) {
        if (!config || !config.enabled) {
            resetHoneypotConfig();
            return;
        }
        
        const hpSwitch = document.getElementById("honeypot-switch");
        const hpOptions = document.getElementById("honeypot-options");
        
        if (hpSwitch) {
            hpSwitch.checked = true;
            hpOptions.style.display = "block";
        }
        
        if (config.seed) {
            document.getElementById("honeypot-seed").value = config.seed;
        }
    }

    function resetHoneypotConfig() {
        const hpSwitch = document.getElementById("honeypot-switch");
        const hpOptions = document.getElementById("honeypot-options");
        const hpSeed = document.getElementById("honeypot-seed");
        
        if (hpSwitch) {
            hpSwitch.checked = false;
        }
        if (hpOptions) {
            hpOptions.style.display = "none";
        }
        if (hpSeed) {
            hpSeed.value = "0.5";
        }
    }

    return {
        initializeEventListeners,
        getHoneypotConfig,
        setHoneypotConfig,
        resetHoneypotConfig
    };
})();

export default HoneypotManager;
