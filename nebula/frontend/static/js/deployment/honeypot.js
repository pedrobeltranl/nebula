// Honeypot Configuration Module
const HoneypotManager = (function () {

    function initializeEventListeners() {
        const hpSwitch = document.getElementById("honeypot-switch");
        const hpOptions = document.getElementById("honeypot-options");

        if (hpSwitch) {
            hpSwitch.addEventListener("change", function () {
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
            seed: parseFloat(document.getElementById("honeypot-seed").value),
            attacker_pivoting: document.getElementById("attacker-pivoting")?.checked || false,
            pivot_round: parseInt(document.getElementById("pivot-round")?.value) || 10,
            global_reset: document.getElementById("honeypot-global-reset")?.checked ?? true
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

        if (document.getElementById("attacker-pivoting")) {
            document.getElementById("attacker-pivoting").checked = config.attacker_pivoting || false;
        }

        if (document.getElementById("honeypot-global-reset")) {
            document.getElementById("honeypot-global-reset").checked = config.global_reset ?? true;
        }
    }

    function resetHoneypotConfig() {
        const hpSwitch = document.getElementById("honeypot-switch");
        const hpOptions = document.getElementById("honeypot-options");
        const hpSeed = document.getElementById("honeypot-seed");
        const hpPivoting = document.getElementById("attacker-pivoting");

        if (hpSwitch) {
            hpSwitch.checked = false;
        }
        if (hpOptions) {
            hpOptions.style.display = "none";
        }
        if (hpSeed) {
            hpSeed.value = "0.5";
        }
        if (hpPivoting) {
            hpPivoting.checked = false;
        }
        if (document.getElementById("honeypot-global-reset")) {
            document.getElementById("honeypot-global-reset").checked = true;
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
